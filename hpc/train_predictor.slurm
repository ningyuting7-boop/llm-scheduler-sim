#!/bin/bash
#SBATCH --job-name=train_predictor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_predictor_%j.out
#SBATCH --error=logs/train_predictor_%j.err

# Phase 2 (docs/Phase2_Predictor_Training_Plan.md): fine-tune DeBERTa-v3-base
# to predict (mu, sigma) of each prompt's log-t output-length distribution,
# from data/qwen3_8b_logt_labels.csv (Phase 1's Qwen3-8B-sampled labels).
#
# GPU model pinned to v100-sxm2 (unlike hpc/generate_labels.slurm's
# gpu:a100:1): an unpinned `--gres=gpu:1` request sat in PD/(Priority) for a
# long time on this cluster's `gpu` partition; ~/projects/if/ifncm's
# slurm/run_eloo.sbatch (same partition, similar mem/time footprint) pins
# v100-sxm2 and schedules quickly, so pinning here too -- not a model
# requirement, just the apparently-less-contended type on this cluster.
#
# Effective batch size is 32 (--batch-size 8 * --grad-accum-steps 4) to
# match the TIE-aligned default (docs/Phase2_Predictor_Training_Plan.md
# section 5) while only ever materializing 8 examples in GPU memory at once
# -- a literal batch_size=32 forward/backward OOM'd on a 32GB GPU during
# interactive testing (deberta-v2's disentangled attention is memory-hungry;
# see conversation notes / plan doc known limitations).
#
# sigma-weight-start/end default to 3.0/4.0 (train_predictor.py's own
# defaults) -- two rounds of testing showed sigma_weight=1.0 (equal) gives
# sigma R^2=0.05, and 2.0 made it WORSE (R^2=0.01), not better. Read TIE's
# own train/model_train.py: they use a 3.0->4.0 linear schedule plus
# gradient clipping (max_norm=1.0, now added to run_epoch) -- much higher
# than either value we'd tried, which is why those attempts underperformed.
# See conversation notes / src/predictor_model.py predictor_loss docstring.
#
# Wall-time: ~2-4h estimated from an interactive smoke test; budgeted 6h
# for margin. Runs via sbatch (not an interactive srun) specifically so it
# survives an SSH/laptop disconnect -- an earlier interactive attempt lost
# an in-progress run to exactly that (best_model.pt up to that point was
# still recovered intact, but training had to restart from epoch 0).

set -euo pipefail

module load anaconda3/2024.06 cuda/12.1.1
source activate /scratch/ning.yuti/envs/tie-repro

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

export HF_HOME=/scratch/ning.yuti/hf_cache

python scripts/train_predictor.py \
    --batch-size "${BATCH_SIZE:-8}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
    --sigma-weight-start "${SIGMA_WEIGHT_START:-3.0}" \
    --sigma-weight-end "${SIGMA_WEIGHT_END:-4.0}" \
    --output-dir "${OUTPUT_DIR:-checkpoints/predictor_frozen_v4}"
