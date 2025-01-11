#!/bin/bash
#SBATCH --job-name=merge_lora           # Job name
#SBATCH --account=mdatascience_team 
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --mem=60G                       # Total memory for the job
#SBATCH --time=02:00:00                 # Maximum time limit (2 hours)
#SBATCH --output=merge_lora_output.log  # Output log file
#SBATCH --error=merge_lora_error.log    # Error log file

# Load necessary modules (adjust based on your environment)

module purge
module load cuda/11.7
source /sw/pkgs/arc/python3.10-anaconda/2023.03/bin/activate
conda activate videollava         # Activate the correct Conda environment
pip install -U transformers
python /home/acapdevi/Video-LLaVA/fine_tuned_models/merge.py
