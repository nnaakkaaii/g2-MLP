# Inductive Node Classification (Graph Neural Networks)

## Author

Yu, Nakai. The University of Tokyo.

Contact : nakai-yu623@g.ecc.u-tokyo.ac.jp

## Results

### PPI (iductive node classification)

| | train accuracy | test accuracy | eclipsed time |
| ---- | ---- | ---- | ---- |
| GAT | 92.96% (±0.24) | 94.17% (±2.20) | 5.0h |
| GCN | 80.81% (±0.13) | 80.74% (±0.69) | 3.2h |

### MUTAG (inductive graph classification)


| | train accuracy | test accuracy | eclipsed time |
| ---- | ---- | ---- | ---- |
| GAT | 83.88% (±2.30) | 83.89% (±9.11) | 0.2h |
| GCN | 85.88% (±1.95) | 82.78% (±9.11) | 0.2h |
| GGAT1 with GAT | 86.12% (±2.41) | 80.56% (±9.70) | 1.6h |
| GGAT1 with GCN | 86.06% (±2.35) | 82.78% (±10.67) | 1.1h |
| GGAT2 with GAT | 84.00% (±1.87) | 80.00 (±11.71) | 2.3h |
| GGAT2 with GCN | 84.82% (±1.60) | 82.78% (±9.77) | 1.5h |

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
