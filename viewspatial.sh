#!/bin/bash
#SBATCH --job-name=viewspatial
#SBATCH -c 1                               # 2 cores
#SBATCH -t 00-02:30:00                     # 6 hours should be enough
#SBATCH -o logs/viewspatial_output_%j.log
#SBATCH -e logs/viewspatial_error_%j.log
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


echo "Starting Viewspatial..."

# Run concatenation script
python viewspatial.py --model qwen3vl_8b_instruct
# python viewspatial.py --model internvl3_5_2b --blind --prompt_type letter-only
# python viewspatial.py --model internvl3_5_2b --blind --prompt_type free-form


echo "Viewspatial completed successfully." 