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
    --gpu_ids 0 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type graph_classification \
    --network_name GNNPool \
    --dataset_name MUTAG \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 100 \
    --data_dir ./inputs/MUTAG/ \
    --index_file_dir ./inputs/MUTAG/10fold_idx/ \
    --name mutag_gat_sagpool \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_gat_sagpool \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GAT \
    --pool_type SAGPool > ./mutag_gat_sagpool.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 1 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type graph_classification \
    --network_name GNNPool \
    --dataset_name MUTAG \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 100 \
    --data_dir ./inputs/MUTAG/ \
    --index_file_dir ./inputs/MUTAG/10fold_idx/ \
    --name mutag_gat_topkpool \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_gat_topkpool \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GAT \
    --pool_type TopKPool > ./mutag_gat_topkpool.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 2 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type graph_classification \
    --network_name GNNPool \
    --dataset_name MUTAG \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 100 \
    --data_dir ./inputs/MUTAG/ \
    --index_file_dir ./inputs/MUTAG/10fold_idx/ \
    --name mutag_gcn_sagpool \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_gcn_sagpool \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GCN \
    --pool_type SAGPool > ./run_mutag_gcn_sagpool.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 3 \
    --verbose \
    --no_visdom_logger \
    --loss_name mce \
    --task_type graph_classification \
    --network_name GNNPool \
    --dataset_name MUTAG \
    --train_transform_name indegree \
    --test_transform_name indegree \
    --optimizer_name adam \
    --n_epochs 100 \
    --data_dir ./inputs/MUTAG/ \
    --index_file_dir ./inputs/MUTAG/10fold_idx/ \
    --name mutag_gcn_topkpool \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_gcn_topkpool \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --gnn_type GCN \
    --pool_type TopKPool > ./mutag_gcn_topkepool.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 4 \
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
    --name mutag_ggat1pool_with_gat \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_ggat1pool_with_gat \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT1 \
    --gnn_type GAT > ./run_mutag_ggat1pool_with_gat.sh.out &
sleep 10
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
    --name mutag_ggat1pool_with_gcn \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_ggat1pool_with_gcn \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT1 \
    --gnn_type GCN > ./run_mutag_ggat1pool_with_gcn.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 6 \
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
    --name mutag_ggat2pool_with_gat \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_ggat2pool_with_gat \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT2 \
    --gnn_type GAT > ./run_mutag_ggat2pool_with_gan.sh.out &
sleep 10
python3 gnn/train.py \
    --gpu_ids 7 \
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
    --name mutag_ggat2pool_with_gcn \
    --save_freq 5 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name debug_mutag_ggat2pool_with_gcn \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --pool_type SAGPool \
    --ggat_type GGAT2 \
    --gnn_type GCN > ./run_mutag_ggat2pool_with_gcn.sh.out &
wait