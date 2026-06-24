#!/bin/bash
#SBATCH --job-name=avse4-eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:3          # Optimized to 1 GPU since evaluation runs sequentially
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=avse4_train_%j.txt

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

echo "Current directory: $(pwd)"
echo "Files:"
ls

module load cuda/12.1

source /users/3128393c/miniconda3/etc/profile.d/conda.sh
conda activate speechenv
export PATH="/users/3128393c/miniconda3/envs/speechenv/bin:$PATH"

# Enable TensorFloat-32 execution for massive speedups on H100 architectures
export NVIDIA_TF32_OVERRIDE=1
export HYDRA_FULL_ERROR=1

echo "Python path: $(which python)"
python --version

echo "Checking CUDA..."
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Starting AVSE4 training..."

# Added the mandatory Hydra overrides so PyTorch Lightning binds seamlessly to your 3 allocated GPUs

srun python train.py \
  trainer.fast_dev_run=False \
  trainer.accelerator=cuda \
  trainer.gpus=3 \
  data.root=/users/3128393c/sharedscratch/data/avsec4/avsec4 \
  trainer.log_dir=/users/3128393c/avse_challenge-main/avse_challenge-main/baseline/avse4/outputs/my_model_run

echo "Job finished at: $(date)"