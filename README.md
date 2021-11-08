# Inductive Node Classification (Graph Neural Networks)

## Author

Yu, Nakai. The University of Tokyo.

Contact : nakai-yu623@g.ecc.u-tokyo.ac.jp

## Results

### PPI (iductive node classification)

| | train accuracy | test accuracy | eclipsed time | run_at |
| ---- | ---- | ---- | ---- | ---- |
| GAT | 92.96% (±0.24) | 94.17% (±2.20) | |
| GCN | 80.81% (±0.13) | 80.74% (±0.69) | h | |
| GGAT1 with GAT | 90.38% (±0.17) | 92.24% (±1.95) | 1.9h (±0.0) | 2021-11-01 08:15 |
| GGAT1 with GCN | 83.83% (±0.24) | 83.58% (±1.22) | h | |
| GGAT2 with GAT | % (±) | % (±) | h | |
| GGAT2 with GCN | 60.09% (±0.04) | 69.05% (±1.18) | h | |
| GGAT1UNet with GAT | 78.11% (±1.44) | 78.30% (±2.49) | h | |
| GGAT1UNet with GCN | 76.83% (±0.23) | 76.85% (±0.65) | h | |
| GGAT2UNet with GAT | % (±) | % (±) | h | |
| GGAT2UNet with GCN | 73.29% (±3.31) | 72.45% (±3.64) | h | |
| UNet with GAT & SAGPool | 84.51% (±0.23) | 86.38% (±1.52) | h | |
| UNet with GAT & TopKPool | 83.47% (±0.17) | 85.09% (±1.54) | h | |
| UNet with GCN & SAGPool | 79.80% (±0.17) | 79.86% (±0.84) | h | |
| UNet with GCN & TopKPool | 79.67% (±0.11) | 79.80% (±0.68) | h | |

### MUTAG (inductive graph classification)

| | train accuracy | test accuracy | eclipsed time | run_at |
| ---- | ---- | ---- | ---- | ---- |
| GAT | 83.88% (±2.30) | 83.89% (±9.11) | 0.2h | |
| GCN | 85.88% (±1.95) | 82.78% (±9.11) | 0.2h | |
| GGAT1 with GAT | 86.12% (±1.85) | 84.44% (±8.53) | 0.2h (±0.0) | 2021-11-01 08:15 |
| GGAT1 with GCN | 87.47% (±1.32) | 81.67% (±8.26) | 0.1h (±0.0) | 2021-11-01 08:15 |
| GGAT2 with GAT | 83.88% (±1.35) | 84.44% (±9.87) | 0.2h (±0.0) | 2021-11-01 08:15 |
| GGAT2 with GCN | 83.88% (±1.60) | 84.44% (±10.18) | 0.1h (±0.0) | 2021-11-01 08:15 |
| GGAT1Pool with GAT | % (±) | % (±) | h | |
| GGAT1Pool with GCN | % (±) | % (±) | h | |
| GGAT2Pool with GAT | % (±) | % (±) | h | |
| GGAT2Pool with GCN | % (±) | % (±) | h | |
| GAT SAGPool | % (±) | % (±) | h | |
| GAT TopKPool | % (±) | % (±) | h | |
| GCN SAGPool | % (±) | % (±) | h | |
| GCN TopKPool | % (±) | % (±) | h | |

## Dataset

| Dataset | PPI |
| ---- | ---- |
| Graphs | 24 |
| Average Nodes Per Graph | 2373 |
| Average Edges Per Graph | 34113 |
| Features of Nodes | 50 |
| Classes | 121 (multilabel) |

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

### Train & View results on Visdom

- PPI x GAT
	```bash
	$ ./scripts/ppi_gat.sh
	```

## Troubleshooting

### `libcudart.so.9.0: cannot open shared object file: No such file or directory`

- pytorch-geometricのバージョンをドキュメントに従って揃える
	- https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- nvidia-toolkitのインストール
	- https://developer.nvidia.com/cuda-downloads
- 環境変数の設定
	- https://stackoverflow.com/questions/58127401/libcudart-so-9-0-cannot-open-shared-object-file-no-such-file-or-directory


### `libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory`

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

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch-Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [DGCNN implementation by leftthomas](https://github.com/leftthomas/DGCNN)
