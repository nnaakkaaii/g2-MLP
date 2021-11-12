#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/g2-MLP/.venv/bin/activate

# n_layers = 16
python3 gnn/train.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name bce \
    --network_name gmlp \
    --dataset_name ppi \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 1000 \
    --data_dir ./inputs/PPI/ \
    --name gmlp_ppi_tuned \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_16 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 128 \
    --ffn_dim 2084 \
    --n_layers 16 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 0.8 &

wait