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
    --loss_name mse \
    --network_name gmlp_node \
    --dataset_name fem_stress_reg \
    --train_transform_name pos_as_attr_label_normalize \
    --val_transform_name pos_as_attr_label_normalize \
    --mean 0.4741 0.6353 -0.0014 \
    --std 12.5448 8.8918 0.7597 \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 1000 \
    --data_dir ./inputs/FEM_STRESS_REG/ \
    --name gmlp_node_fem_stress_reg \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1000 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 128 \
    --ffn_dim 2084 \
    --n_layers 20 \
    --lr_decay_iters 200 \
    --lr_decay_gamma 0.3 \
    --prob_survival 0.8 \
    --regression

python3 gnn/inference.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 1024 \
    --verbose \
    --network_name gmlp_node \
    --dataset_name fem_stress_reg \
    --name gmlp_node_fem_stress_reg \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1000