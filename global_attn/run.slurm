#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --output=/home/wolf1/davidhay/global_attn/wiki2-4layer-8iterations-256window.txt # redirect stdout and stderr
#SBATCH --partition=gpu-a100-killable # (see resources section) gpu-a100-killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gres=gpu:3
#SBATCH --constraint="a100"

PYTHONUNBUFFERED=1 HF_HOME="/home/wolf1/davidhay/.cache" accelerate launch --main_process_port=29500 train.py --bptt=256 --batch_size=4 --eval_batch_size=1 --dataset="wiki2" --nlayers=4 --lr=1e-4 --k=8
