#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

# n_layers = 20
python3 gnn/train.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 1024 \
    --verbose \
    --loss_name mce \
    --network_name mlp_node_classification \
    --dataset_name fem \
    --train_transform_name pos_all \
    --val_transform_name pos_all \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 1500 \
    --data_dir ./inputs/FEM/ \
    --name mlp_node_classification_fem \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1500 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 128 \
    --ffn_dim 2084 \
    --n_layers 20 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 0.8 &

wait