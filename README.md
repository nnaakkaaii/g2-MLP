# g2-MLP

## Results

### PPI (iductive node classification)

| Model | micro-F1 | details |
| ---- | ---- | ---- |
| GAT | 94.17% (±2.20) | |
| GCN | 80.74% (±0.69) | |
| g2-MLP (16 layers, pb 0.8, 1000 epochs) | 99.53% (±0.01) | 99.52<br>99.53<br>99.51<br>99.54<br>99.53 |
| g2-MLP (20 layers, pb 0.8, 1500 epochs) | 99.54% (±0.01) | 99.55<br>99.54<br>99.56<br>99.52<br>99.54 |
| g2-MLP (24 layers, pb 0.8, 1500 epochs) | 99.53% (±0.01) | 99.53<br>99.52<br>99.54<br>99.52<br>99.54 |
| g2-MLP (20 layers, pb 0.6, 1500 epochs) | 99.49% (±0.02) | 99.48<br>99.52<br>99.51<br>99.48<br>99.48 |
| g2-MLP (20 layers, pb 1.0, 1500 epochs) | 99.41% (±0.02) | 99.44<br>99.39<br>99.38<br>99.42<br>99.40 |

<details>
<summary>ハイパラ詳細</summary>
<div>

| parameters | value |
| ---- | ---- |
| batch size | 64 |
| lr | 2.5e-3 |
| beta | (0.9, 0.9) |
| lr_decay_gamma | 0.3 |
| lr_decay_iters | 300 |
| fnn hidden dim | 2048 |
| hidden dim | 128 |

</div>
</details>


### NCI1 (inductive graph classification)

| Model | Accuracy | details |
| ---- | ---- | ---- |
| SAGPool | 74.18% |
| GIN-0 | 82.7% |
| PSCN | 78.59% |
| GK | 62.28 |
| g2-MLP (4 layers, pb 1.0, 100 epochs) | 82.38% (±0.76) | 82.48<br>83.21<br>83.21<br>81.51<br>81.51 |

<details>
<summary>ハイパラ詳細</summary>
<div>

| parameters | value |
| ---- | ---- |
| batch size | 256 |
| lr | 2.5e-3 |
| beta | (0.9, 0.9) |
| fnn hidden dim | 512 |
| hidden dim | 32 |

</div>
</details>


### PTC_MR (inductive graph classification)

| Model | Accuracy | details |
| ---- | ---- | ---- |
| U2GNN | 69.93% | |
| GAT | 66.70% | |
| GIN-0 | 64.6% | |
| PSCN | 62.29% | |
| GK | 57.26% | |
| g2-MLP (4 layers, 50 epochs) | 68.00% (±2.14) | 71.43<br>68.57<br>65.71<br>68.57<br>65.71 |

<details>
<summary>ハイパラ詳細</summary>
<div>

| parameters | value |
| ---- | ---- |
| batch size | 2048 |
| lr | 1.18e-4 |
| beta | (0.9, 0.9) |
| fnn hidden dim | 1024 |
| hidden dim | 128 |

</div>
</details>


### PROTEINS (inductive graph classification)

| Model | Accuracy | details |
| ---- | ---- | ---- |
| U2GNN | 78.53% | |
| SAGPool | 71.86% | |
| GCN | 75.65% | |
| GAT | 74.70% | |
| GIN-0 | 76.2% | |
| PSCN | 75.89% | |
| GK | 71.67% | |

## Dataset

| Dataset | PPI | NCI1 | PTC_MR | PROTEINS |
| ---- | ---- | ---- | ---- | ---- |
| Graphs | 24 | 4110 | 344 | 1113 |
| Average Nodes Per Graph | 2373 | 29.87 | 14.29 | 39.06 |
| Average Edges Per Graph | 34113 | 32.30 |14.69 | 72.82 |
| Features of Nodes | 50 | | | |
| Classes | 121 (multilabel) | 2 | 2 | 2 |

## Model

### Graph Attention Networks (GAT; 2017)

Reference : [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

### Graph Convolution Networks (GCN; 2016)

Reference : [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

## Preparation

1. install dependent packages

	```bash
	$ pip3 install torch==1.9.1
	$ pip3 install -r requirements.txt
	```

2. download necessary data

	```bash
	$ make
	```

## Usage

### train

```bash
$ ./scripts/ppi_gat.sh
```

### view tuning result on optuna-dashboard

```bash
$ optuna-dashboard sqlite:///db.sqlite3
```



## Troubleshooting

<details><summary> `libcudart.so.9.0: cannot open shared object file: No such file or directory` </summary>
<div>

- pytorch-geometricのバージョンをドキュメントに従って揃える
	- https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- nvidia-toolkitのインストール
	- https://developer.nvidia.com/cuda-downloads
- 環境変数の設定
	- https://stackoverflow.com/questions/58127401/libcudart-so-9-0-cannot-open-shared-object-file-no-such-file-or-directory

</div>
</details>


<details><summary> `libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory` </summary>
<div>

- pytorchのバージョンがあっていない
- torch==1.9.1をインストール後、案内に従って残りのライブラリをインストール

```bash
$ python -c "import torch; print(torch.__version__)"
1.9.1
$ python -c "import torch; print(torch.version.cuda)"
cu10.2
$ export TORCH=1.9.1
$ export CUDA=cu102
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-geometric
```

</div>
</details>

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch-Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [DGCNN implementation by leftthomas](https://github.com/leftthomas/DGCNN)
- [HGP-SL implementation by cszhangzhen](https://github.com/cszhangzhen/HGP-SL)

## Author

Yu, Nakai. The University of Tokyo.

Contact : nakai-yu623@g.ecc.u-tokyo.ac.jp

