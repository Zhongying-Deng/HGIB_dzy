#!/bin/bash
#SBATCH --account KOURTZI-SL2-GPU
#SBATCH -p ampere
#SBATCH -c 120
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#! SBATCH --exclusive

### srun -p ampere -t 24:00:00 --nodes=1 --gpus-per-node=1 -A CIA-DAMTP-SL2-GPU --pty bash


conda activate HGIB

python train.py --gpu_ids 0, --lr 0.0001 --name HGIB/your_name --netD fc --focal --model single_time_multi_modality_classification_HGIB --label_time m24 --onefc --control MCI --K_neigs 20 --continue_train --load_weight non_image/complete-modality-info10 --niter 0 --niter_decay 2000  --beta 10 --split 10
python test.py --gpu_ids 0, --name HGIB --netD fc --focal --model single_time_multi_modality_classification_HGIB --label_time m24 --onefc --control MCI --K_neigs 20 --load_weight HGIB/your_name --split 10
