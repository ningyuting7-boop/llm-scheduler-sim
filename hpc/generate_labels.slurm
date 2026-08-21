#!/bin/bash
#SBATCH --job-name=gen_predictor_labels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/gen_predictor_labels_%j.out
#SBATCH --error=logs/gen_predictor_labels_%j.err

# Phase 1 (docs/VLLM_Qwen3_Reproduction_Plan.md): generate TIE predictor
# training labels from real Qwen3-8B completions on LMSYS-Chat-1M prompts.
#
# Single job, continuing from the 3000-prompt pilot already generated
# (data/qwen3_8b_logt_labels.csv / qwen3_8b_length_samples.csv) --
# start-index skips those so we don't regenerate them. Writes to
# separately-named "_extra" files rather than overwriting/appending to
# the pilot files in place, so combining the two afterwards is a simple
# one-off `cat` with no risk of double-counting on a re-run (see plan
# doc discussion of the merge-idempotency issue we deliberately avoided
# by not building a repeatable N-shard merge for this single-job path).
#
# Wall-time: ~2h got through the 3000-prompt pilot, so 7000 more prompts
# is budgeted at ~8h with margin; adjust PROMPTS_PER_SHARD/--time if you
# change the scale.

set -euo pipefail

module load anaconda3/2024.06
source activate /scratch/ning.yuti/envs/tie-repro

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

export HF_HOME=/scratch/ning.yuti/hf_cache
export VLLM_CACHE_ROOT=/scratch/ning.yuti/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/ning.yuti/torchinductor_cache

python scripts/generate_predictor_labels.py \
    --start-index "${START_INDEX:-3000}" \
    --num-prompts "${NUM_PROMPTS:-7000}" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT:-20}" \
    --labels-out data/qwen3_8b_logt_labels_extra.csv \
    --samples-out data/qwen3_8b_length_samples_extra.csv
