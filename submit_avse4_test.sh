#!/bin/bash
#SBATCH --job-name=avse4-eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=avse4_test_%j.txt
set -e

# --- CONFIGURATION VARIABLES ---
TARGET_VERSION="version_28"
# -------------------------------

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

BASE_DIR="/users/3128393c/avse_challenge-main/avse_challenge-main/baseline/avse4"

# 1. Force-wipe the correct destination folder to reset the evaluation file guard
echo "Clearing out previous evaluation outputs for ${TARGET_VERSION}..."
rm -rf "${BASE_DIR}/outputs/my_model_run/enhanced_outputs/${TARGET_VERSION}"

module load cuda/12.1
source /users/3128393c/miniconda3/etc/profile.d/conda.sh
conda activate speechenv
export PATH="/users/3128393c/miniconda3/envs/speechenv/bin:$PATH"

export NVIDIA_TF32_OVERRIDE=1
export HYDRA_FULL_ERROR=1

# 2. Automatically find the checkpoint file in the target version directory
CKPT_FOLDER="${BASE_DIR}/outputs/my_model_run/lightning_logs/${TARGET_VERSION}/checkpoints"

# Verify the folder actually exists
if [ ! -d "$CKPT_FOLDER" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_FOLDER"
    exit 1
fi

# Find the first .ckpt file in that directory automatically
RAW_CKPT_PATH=$(find "$CKPT_FOLDER" -maxdepth 1 -name "*.ckpt" ! -name "eval_target.ckpt" | head -n 1)

if [ -z "$RAW_CKPT_PATH" ]; then
    echo "ERROR: No checkpoint (.ckpt) file found in $CKPT_FOLDER"
    exit 1
fi

CLEAN_CKPT_PATH="${CKPT_FOLDER}/eval_target.ckpt"
echo "Found checkpoint: $RAW_CKPT_PATH"
echo "Creating clean string reference for Hydra..."
ln -sf "$RAW_CKPT_PATH" "$CLEAN_CKPT_PATH"

echo "Starting AVSE4 Evaluation and Objective Metric Compilation..."

# 3. Execute testing script using the sanitized path link
python test.py \
  hydra.job.chdir=False \
  data.root=/users/3128393c/sharedscratch/data/avsec4/avsec4 \
  ckpt_path="${CLEAN_CKPT_PATH}" \
  data.dev_set=True \
  data.eval_set=False \
  save_dir="${BASE_DIR}/outputs/my_model_run/enhanced_outputs" \
  model_uid="${TARGET_VERSION}"

echo "Job finished at: $(date)"