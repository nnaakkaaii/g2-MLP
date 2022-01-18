#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

python3 gnn/train.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name bce \
    --network_name gcn_node \
    --dataset_name ppi \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 300 \
    --data_dir ./inputs/PPI/ \
    --name gcn_node_ppi_final \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_8_300 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 512 \
    --n_layers 8 \
    --dropout_rate 0.1 \
    --lr_decay_iters 90 \
    --lr_decay_gamma 0.3 &

wait