#!/bin/sh

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gs84
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source /work/02/gs84/s84000/inductive_node_classification_models/.venv/bin/activate

# dd
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/DD/ \
    --name tuning_gmlp_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name dd \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/DD/ \
    --name tuning_gmlp_graph_classification_dd \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
# enzymes
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name enzymes \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/ENZYMES/ \
    --name tuning_gmlp_graph_classification_enzymes \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name enzymes \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/ENZYMES/ \
    --name tuning_gmlp_graph_classification_enzymes \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
# frankenstein
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name frankenstein \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/FRANKENSTEIN/ \
    --name tuning_gmlp_graph_classification_frankenstein \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name frankenstein \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/FRANKENSTEIN/ \
    --name tuning_gmlp_graph_classification_frankenstein \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
# nci1
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/NCI1/ \
    --name tuning_gmlp_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name nci1 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/NCI1/ \
    --name tuning_gmlp_graph_classification_nci1 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
# nci109
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name nci109 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/NCI109/ \
    --name tuning_gmlp_graph_classification_nci109 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name nci109 \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/NCI109/ \
    --name tuning_gmlp_graph_classification_nci109 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
# proteins
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name proteins \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/PROTEINS/ \
    --name tuning_gmlp_graph_classification_proteins \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &
python3 gnn/tuning.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --batch_size 64 \
    --verbose \
    --loss_name mce \
    --network_name gmlp_graph_classification \
    --dataset_name proteins \
    --train_transform_name indegree \
    --val_transform_name indegree \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 150 \
    --data_dir ./inputs/PROTEINS/ \
    --name tuning_gmlp_graph_classification_proteins \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name tuning_1115_0437 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 32 \
    --ffn_dim 512 \
    --n_layers 4 \
    --lr_decay_iters 300 \
    --lr_decay_gamma 0.3 \
    --prob_survival 1. &

wait