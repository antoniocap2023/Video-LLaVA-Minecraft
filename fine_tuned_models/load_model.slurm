#!/bin/bash

# Slurm directives
#SBATCH --job-name=video_llava_save     # Give your job a descriptive name
#SBATCH --account=engin1               # Use your account
#SBATCH --partition=gpu
#SBATCH --time=00:30:00                # Allow enough time
#SBATCH --nodes=1                      # One node
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=2              # Two CPU cores for data handling
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=32G                      # Allocate 32G memory
#SBATCH --output=video_llava_save.%j.out  # Save standard output to file
#SBATCH --error=video_llava_save.%j.err   # Save error logs to file

# Activate environment
conda activate videollava

# Run Python file
python /home/acapdevi/Video-LLaVA/models/pretrain_model.py
