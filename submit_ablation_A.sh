#!/bin/bash
#SBATCH --job-name=abl-freq-warp
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=ablation_A_%j.txt
#SBATCH --requeue

# Experiment A: FrequencyWarp -- learnable per-frequency power-law compression
# applied before the encoder. 257 extra parameters, negligible compute.
# See abalation_readme.md for hypothesis and reporting requirements.

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

cd "$SLURM_SUBMIT_DIR"

module load cuda/12.1
source /users/3128393c/miniconda3/etc/profile.d/conda.sh
conda activate speechenv
export PATH="/users/3128393c/miniconda3/envs/speechenv/bin:$PATH"

export NVIDIA_TF32_OVERRIDE=1
export HYDRA_FULL_ERROR=1
export ABLATION_FREQ_WARP=true
export ABLATION_FREQ_MOD=false

echo "Ablation: FREQ_WARP=true  FREQ_MOD=false"
python --version
python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0))"

BASE_DIR="/users/3128393c/avse_challenge-main/avse_challenge-main/baseline/avse4"
LOG_DIR="${BASE_DIR}/outputs/ablation_A"

# Auto-resume from this experiment's own last.ckpt (never from main runs)
LAST_CKPT=$(find "${LOG_DIR}/lightning_logs" -name "last.ckpt" \
            -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from: $LAST_CKPT"
    CKPT_ARG="trainer.ckpt_path=${LAST_CKPT}"
else
    echo "Starting Experiment A from scratch."
    CKPT_ARG="trainer.ckpt_path=null"
fi

echo "Starting Experiment A training..."

srun python train_ablation.py \
  trainer.fast_dev_run=False \
  trainer.accelerator=cuda \
  trainer.gpus=3 \
  data.root=/users/3128393c/sharedscratch/data/avsec4/avsec4 \
  trainer.log_dir=${LOG_DIR} \
  "${CKPT_ARG}"

echo "Job finished at: $(date)"
