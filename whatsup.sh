#!/bin/bash
#SBATCH --job-name=whatsup
#SBATCH -c 1                               # 2 cores
#SBATCH -t 00-06:00:00                     # 6 hours should be enough
#SBATCH -o logs/whatsup_output_%j.log
#SBATCH -e logs/whatsup_error_%j.log
#SBATCH --open-mode=append
#SBATCH -p seas_gpu
#SBATCH --account=ydu_lab
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate spatial

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 
export HF_DATASETS_CACHE=/net/holy-isilon/ifs/rc_labs/ydu_lab/alex/.cache/huggingface/datasets
export HF_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache/huggingface
export TORCH_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache/torch


echo "Starting Whatsup..."

# Run concatenation script
python whatsup.py --prompt_type direct

echo "Whatsup completed successfully." 