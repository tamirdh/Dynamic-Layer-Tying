#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --output=/home/wolf1/davidhay/global_attn/exp0.txt # redirect stdout and stderr
#SBATCH --partition=gpu-a100-killable # (see resources section) gpu-a100-killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gres=gpu:3
#SBATCH --constraint="a100"

PYTHONUNBUFFERED=1 HF_HOME="/home/wolf1/davidhay/.cache" accelerate launch --main_process_port=29500 train.py --bptt=256 --batch_size=100 --eval_batch_size=1 --dataset="wiki2" --nlayers=12 --lr=1e-4 --emsize=1600 --k=16 --factor=1.0 --exp="Hyper12" --epochs=600
