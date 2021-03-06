python3 gnn/train.py \
    --gpu_ids 0 \
    --batch_size 32 \
    --verbose \
    --loss_name mse \
    --network_name gcn_node \
    --dataset_name fem_disp_reg \
    --train_transform_name pos_as_attr_label_normalize \
    --val_transform_name pos_as_attr_label_normalize \
    --mean -0.0050 -0.0129 0 \
    --std 0.0153 0.0362 0.0002 \
    --optimizer_name adam \
    --scheduler_name step \
    --n_epochs 100 \
    --data_dir ./inputs/FEM_DISP_REG/ \
    --name gcn_node_fem_disp_reg \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_2_512_100 \
    --lr 2.5e-3 \
    --beta1 0.9 \
    --beta2 0.9 \
    --hidden_dim 512 \
    --n_layers 2 \
    --dropout_rate 0.1 \
    --lr_decay_iters 30 \
    --lr_decay_gamma 0.3 \
    --regression

python3 gnn/inference.py \
    --gpu_ids 0 \
    --batch_size 32 \
    --verbose \
    --network_name gcn_node \
    --dataset_name fem_disp_reg \
    --name gcn_node_fem_disp_reg \
    --mlflow_root_dir ./mlruns/ \
    --run_name layer_2_512_100 \
    --save_freq 10 \
    --save_dir ./checkpoints \
    --regression