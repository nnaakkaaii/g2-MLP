#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

# COLLAB (5000 x 74.5)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 256 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name COLLAB \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/COLLAB/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_collab \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1.

# IMDB-BINARY (1000 x 19.8)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 128 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name IMDB-BINARY \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/IMDB-BINARY/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_imdb_binary \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1.

# IMDB-MULTI (1500 x 13)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 128 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name IMDB-MULTI \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/IMDB-MULTI/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_imdb_multi \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1.

# DD (1178 x 284.3)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 128 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name DD \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/DD/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1.

# PROTEINS (1113 x 39.1)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 128 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name PROTEINS \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/PROTEINS/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_proteins \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1.

# NCI1 (4110 x 29.9)
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 256 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_hierarchical_sagpool_graph_classification \
    --dataset_name NCI1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/NCI1/ \
    --name tuning_gmlp_hierarchical_sagpool_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1118_0037 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \

wait