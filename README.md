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


### FEM (inductive node classification)

| Model | micro-F1 | details |
| ---- | ---- | ---- |
| GAT (4 layers, 512 hidden_dim, 100 epochs) | 39.56% (±0.04) | 39.55<br>39.53<br>39.63<br>39.55<br>39.52 |
| GCN (4 layers, 512 hidden_dim, 100 epochs) | 40.60% (±0.17) | 40.69<br>40.72<br>40.70<br>40.27<br>40.62 |
| MLP (12 layers, pb 0.8, 1500 epochs) | % (±) | <br><br><br><br> |
| g2-MLP (12 layers, pb 0.8, 1500 epochs) | % (±) | <br><br><br><br> |

<details>
<summary>ハイパラ詳細</summary>
<div>

| parameters | value |
| ---- | ---- |
| batch size | 512 |
| lr | 2.5e-3 |
| beta | (0.9, 0.9) |
| lr_decay_gamma | 0.3 |
| lr_decay_iters | 300 |
| fnn hidden dim | 2048 |
| hidden dim | 128 |

</div>
</details>

<details>
<summary>データセット詳細</summary>
<div>

次のようなフィレット構造を対象とする。

![fillet](./docs/fillet.png)

次のようなパターンに対して、 229 個のデータを用意した。

- 各長方形の高さをそれぞれ 10 ~ 100 (10刻み) でランダムに変更
- フィレット径を 5 ~ 45 (5刻み) でランダムに変更

このうち、8割をtrain, 1割をvalidation, 1割をtestとした

./docs/FEMMeshNetgen.frd にあるようなCalculixの計算結果をパースした.

特徴量には次のDisp. を利用している. (TODO : dispとは?)

```
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1-1.23851E-03 1.38397E-03 8.13754E-06
 -1         2-1.23833E-03 1.38748E-03 1.65606E-05
 -1         3-1.23595E-03 1.86246E-03 1.33735E-05
 ...
```

また、次のノードの座標も取得し、正規化した上で特徴量として取り入れている.

```
    2C                          1242                                     1
 -1         1 2.30000E+01 2.00000E+01 5.00000E+00
 -1         2 2.30000E+01 2.00000E+01 0.00000E+00
 -1         3 3.00000E+01 2.00000E+01 5.00000E+00
 -1         4 3.00000E+01 2.00000E+01 0.00000E+00
 -1         5 3.00000E+01 0.00000E+00 5.00000E+00
```

予測対象は次の応力Stress. を、応力の大きさ ( $\sqrt(SXX^2 + SYY^2 + SZZ^2)$ ) にしたものを利用している.

また、予測は数値を直接予測するのではなく、 $2^n$ でbinningし (今回は $2^0$ から $2^4$ )、クラス分類のタスクとした.

(ex : `[0.1, 7, 3, 18, 2]` -> `[0, 3, 2, 5, 1]`)

```
 -4  STRESS      6    1
 -5  SXX         1    4    1    1
 -5  SYY         1    4    2    2
 -5  SZZ         1    4    3    3
 -5  SXY         1    4    1    2
 -5  SYZ         1    4    2    3
 -5  SZX         1    4    3    1
 -1         1 3.68768E-01 8.27878E-03-2.20199E-02-1.50879E-01 2.03513E-03 3.27718E-02
 -1         2 3.52119E-01 1.33049E-02-2.21362E-02-1.71304E-01 8.95834E-03-4.10380E-02
 -1         3 3.28198E-03-6.22315E-03 1.52381E-03-3.40027E-03-1.19905E-03-2.91369E-04
 -1         4-3.29518E-03-8.83664E-03 1.10334E-03 3.01675E-03 7.87600E-04 5.29066E-04
```

ノード間の結合は、次のような正四面体要素であることを踏まえて、正四面体構成ノードのrawデータからパースした

![tetrahedron](https://wiki.freecadweb.org/images/7/70/FEM_mesh_elements_4_tetrahedron.svg)

```
    3C                           605                                     1
 -1       483    6    0    1
 -2        23       142        24        52       810       811       228       793       525       797
 -1       484    6    0    1
 -2       142        80        24        52       812       785       811       525       795       797
 -1       485    6    0    1
 -2       119       128       180       181       469       813       815       816       814       659
```

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

