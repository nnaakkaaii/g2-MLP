#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

# source /work/gs84/s84000/inductive_node_classification_models/.venv/bin/activate
source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate
# source /work/opt/local/x86_64/apps/cuda/11.1/pytorch/1.8.1/bin/activate
python3 gnn/train.py \
    --gpu_ids 0 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name GGATUNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name ggat1unet_with_gat_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_ggat1unet_with_gat_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT1 \
    --gnn_type GAT &
sleep 10
python3 gnn/train.py \
    --gpu_ids 1 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name GGATUNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name ggat1unet_with_gcn_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_ggat1unet_with_gcn_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT1 \
    --gnn_type GCN
sleep 10
python3 gnn/train.py \
    --gpu_ids 2 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name GGATUNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name ggat2unet_with_gat_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_ggat2unet_with_gat_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT2 \
    --gnn_type GAT
sleep 10
python3 gnn/train.py \
    --gpu_ids 3 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name GGATUNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name ggat2unet_with_gcn_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_ggat2unet_with_gcn_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT2 \
    --gnn_type GCN
sleep 10
python3 gnn/train.py \
    --gpu_ids 4 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name UNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name unet_with_gat_sagpool_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_unet_with_gat_sagpool_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GAT \
    --pool_type SAGPool
sleep 10
python3 gnn/train.py \
    --gpu_ids 5 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name UNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name unet_with_gat_topkpool_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_unet_with_gat_topkpool_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GAT \
    --pool_type TopKPool
sleep 10
python3 gnn/train.py \
    --gpu_ids 6 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name UNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name unet_with_gcn_sagpool_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_unet_with_gcn_sagpool_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GCN \
    --pool_type SAGPool
sleep 10
python3 gnn/train.py \
    --gpu_ids 7 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type multi_label_node_classification \
    --network_name UNet \
    --dataset_name PPI \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 500 \
    --data_dir ./inputs/PPI/ \
    --index_file_dir ./inputs/PPI/10fold_idx/ \
    --name unet_with_gcn_topkpool_ppi \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_unet_with_gcn_topkpool_ppi \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GCN \
    --pool_type TopKPool