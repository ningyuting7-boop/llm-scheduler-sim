#!/bin/bash
#SBATCH --job-name=gen_predictor_labels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_predictor_labels_%j.out
#SBATCH --error=logs/gen_predictor_labels_%j.err

# Phase 1 (docs/VLLM_Qwen3_Reproduction_Plan.md): generate TIE predictor
# training labels from real Qwen3-8B completions on LMSYS-Chat-1M prompts.
#
# --gres=gpu:a100:1 matches the A100-SXM4-80GB this pipeline was sanity
# checked on; swap to gpu:h200:1 if you'd rather use an H200 (also fine,
# just untested by us specifically). Wall-time is a rough estimate for the
# ~3000-prompt pilot scale (see plan doc step 5) -- adjust once you've seen
# real throughput on your allocation.

set -euo pipefail

module load anaconda3/2024.06
source activate /scratch/ning.yuti/envs/tie-repro

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

export HF_HOME=/scratch/ning.yuti/hf_cache

python scripts/generate_predictor_labels.py \
    --num-prompts "${NUM_PROMPTS:-3000}" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT:-20}"
