#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

# DD (1178 x 284.3)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 2048 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/DD/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 2048 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/DD/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 2048 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/DD/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 2048 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/DD/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &

# NCI1 (4110 x 29.9)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 5096 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/NCI1/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 5096 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/NCI1/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 5096 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/NCI1/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 5096 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/NCI1/ \
    --name 1118_tuning_gmlp_hierarchical_sagpool_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0135 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 &

sleep 180 && nvidia-smi &

wait