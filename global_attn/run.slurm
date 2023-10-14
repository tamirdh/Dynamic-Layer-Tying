#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --output=/home/wolf1/davidhay/global_attn/llama2-full.txt # redirect stdout and stderr
#SBATCH --partition=gpu-a100-killable # (see resources section) gpu-a100-killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gres=gpu:3
#SBATCH --constraint="a100"

PYTHONUNBUFFERED=1 NCCL_SOCKET_TIMEOUT="10800s" HF_HOME="/home/wolf1/davidhay/.cache" accelerate launch --main_process_port=29507 train.py --bptt=256 --batch_size=16 --eval_batch_size=1 --dataset="wiki2" --nlayers=48 --lr=1e-4 --emsize=1600 --exp="llama2-full" --epochs=300 --llama
