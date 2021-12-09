#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

pip3 install matplotlib

python3 gnn/inference.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 1024 \
    --verbose \
    --network_name gmlp_node \
    --dataset_name fem_stress_cls \
    --name gmlp_node_fem_stress_cls \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1500


python3 gnn/inference.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 1024 \
    --verbose \
    --network_name mlp_node \
    --dataset_name fem_stress_cls \
    --name mlp_node_fem_stress_cls \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1500


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
    --run_name layer_20_1500


python3 gnn/inference.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 1024 \
    --verbose \
    --network_name mlp_node \
    --dataset_name fem_stress_reg \
    --name mlp_node_fem_stress_reg \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_20_1500