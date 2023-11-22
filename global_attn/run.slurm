#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --output=/home/wolf1/davidhay/global_attn/abl-rev.txt # redirect stdout and stderr
#SBATCH --partition=gpu-a100-killable # (see resources section) gpu-a100-killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

PYTHONUNBUFFERED=1 HF_HOME="/home/wolf1/davidhay/.cache" accelerate launch --main_process_port=29505 dqn_trainer.py --bptt=256 --batch_size=16 --eval_batch_size=1 --dataset="shake" --nlayers=12 --lr=1e-4 --emsize=1600 --nhead=16 --exp="abl-rev" --epochs=300
