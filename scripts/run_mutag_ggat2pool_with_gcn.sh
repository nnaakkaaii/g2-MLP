#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=10:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

# source /work/gs84/s84000/inductive_node_classification_models/.venv/bin/activate
source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate
# source /work/opt/local/x86_64/apps/cuda/11.1/pytorch/1.8.1/bin/activate
python3 gnn/train.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type graph_classification \
    --network_name GGATPool \
    --dataset_name MUTAG \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 100 \
    --data_dir ./inputs/MUTAG/ \
    --index_file_dir ./inputs/MUTAG/10fold_idx/ \
    --name ggat2pool_with_gcn_mutag \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_ggat2pool_with_gcn_mutag \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT2 \
    --gnn_type GCN
