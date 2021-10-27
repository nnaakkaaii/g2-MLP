# Various Graph Neural Network Models

## Supported

- Dataset
	- CoraDataset
- Model
	- Graph Attention Networks


## Errors

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