#!/bin/bash
#SBATCH --job-name=tie_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/tie_smoke_%j.out
#SBATCH --error=logs/tie_smoke_%j.err

# Batch wrapper around scripts/smoke_test.sh (phase 3 task 9).
#
# Submitted with sbatch rather than run under an interactive srun because the
# smoke test is self-contained -- it starts the server, pushes a burst,
# prints a verdict, and cleans up -- so interactivity buys nothing, while an
# SSH drop would tear down an srun allocation mid-run. Phase 2 lost a
# training run to exactly that.
#
#   sbatch hpc/smoke_test.slurm                      # tie arm, defaults
#   sbatch --export=ALL,ARM=fcfs_pred hpc/smoke_test.slurm
#   sbatch --export=ALL,ARM=tie,RATE=64 hpc/smoke_test.slurm
#
# Watch it with:  tail -f logs/tie_smoke_<jobid>.out

set -uo pipefail

module load anaconda3/2024.06 cuda/12.1.1
source activate /scratch/ning.yuti/envs/tie-repro

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export HF_HOME=/scratch/ning.yuti/hf_cache

ARM="${ARM:-tie}"
NUM_PROMPTS="${NUM_PROMPTS:-60}"
RATE="${RATE:-32}"

echo "job $SLURM_JOB_ID on $(hostname)"
bash scripts/smoke_test.sh "$ARM" "$NUM_PROMPTS" "$RATE"
