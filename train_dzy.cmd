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
# the best result
python train_ssl.py --gpu_ids 0, --lr 0.0001 --name HGIB/dzy_wu01 --netD fc --focal --model single_time_multi_modality_classification_HGIB_SS --label_time m24 --onefc --control MCI --K_neigs 20 --continue_train --load_weight non_image/complete-modality-info10 --niter 0 --niter_decay 2000  --beta 10 --split 10 --num_graph_update 100 --save_latest_freq 100 --save_epoch_freq 20 --weight_u 0.1 2>&1 | tee checkpoints/log-HGIB-unlabeled-cons-V1Rerun-new.log
train_ssl_V1.py --gpu_ids 0, --lr 0.0001 --name HGIB/ssl --netD fc --focal --model HGIB_semi_unlabeled_consistency --label_time m24 --onefc --control MCI --K_neigs 20 --continue_train --load_weight non_image/complete-modality-info10 --niter 0 --niter_decay 2000  --beta 10 --split 10 --num_graph_update 100 --save_latest_freq 100 --save_epoch_freq 20 --weight_u 0.1 --use_strong_aug --train_encoders
