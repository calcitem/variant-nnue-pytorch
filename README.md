# Variant NNUE trainer

This is the chess variant NNUE training code for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish). See the documentation in the [wiki](https://github.com/ianfab/variant-nnue-pytorch/wiki) for more details on the training process and [join our discord](https://discord.gg/FYUGgmCFB4) to ask questions. This project is derived from the [trainer for standard chess](https://github.com/glinscott/nnue-pytorch) used for official Stockfish.

# Setup

#### Install PyTorch

[PyTorch installation guide](https://pytorch.org/get-started/locally/)

```
python3 -m venv env
source env/bin/activate
pip install python-chess==0.31.4 pytorch-lightning<1.5.0 torch matplotlib
```

#### Install CuPy
First check what version of cuda is being used by pytorch.
```
import torch
torch.version.cuda
```
Then install CuPy with the matching CUDA version.
```
pip install cupy-cudaXXX
```
where XXX corresponds to the first 3 digits of the CUDA version. For example `cupy-cuda112` for CUDA 11.2.

CuPy might use the PyTorch's private installation of CUDA, but it is better to install the matching version of CUDA separately. [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

#### Build the fast DataLoader
This requires a C++17 compiler and cmake.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

## Resuming from a checkpoint
```
python train.py --resume_from_checkpoint <path> ...
```

## Training on GPU
```
python train.py --gpus 1 ...
```
## Feature set selection
By default the trainer uses a factorized HalfKAv2 feature set (named "HalfKAv2^")
If you wish to change the feature set used then you can use the `--features=NAME` option. For the list of available features see `--help`
The default is:
```
python train.py ... --features="HalfKAv2^"
```

## Skipping certain fens in the training

`--smart-fen-skipping` currently skips over moves where the king is in check, or where the bestMove is a capture (typical of non-quiet positions).
`--random-fen-skipping N` skip N fens on average before using one. Uses fewer fens per game, useful with large data sets.

## Current recommended training invocation

```
python train.py --smart-fen-skipping --random-fen-skipping 3 --batch-size 16384 --threads 8 --num-workers 8 --gpus 1 trainingdata validationdata
```
best nets have been trained with 16B d9-scored nets, training runs >200 epochs



# Export a network

Using either a checkpoint (`.ckpt`) or serialized model (`.pt`),
you can export to SF NNUE format.  This will convert `last.ckpt`
to `nn.nnue`, which you can load directly in SF.
```
python serialize.py last.ckpt nn.nnue
```

# Import a network

Import an existing SF NNUE network to the pytorch network format.
```
python serialize.py nn.nnue converted.pt
```

# Visualize a network

Visualize a network from either a checkpoint (`.ckpt`), a serialized model (`.pt`)
or a SF NNUE file (`.nnue`).
```
python visualize.py nn.nnue --features="HalfKAv2"
```

Visualize the difference between two networks from either a checkpoint (`.ckpt`), a serialized model (`.pt`)
or a SF NNUE file (`.nnue`).
```
python visualize.py nn.nnue  --features="HalfKAv2" --ref-model nn.cpkt --ref-features="HalfKAv2^"
```

# Logging

```
pip install tensorboard
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

# Automatically run matches to determine the best net generated by a (running) training

```
python run_games.py --concurrency 16 --stockfish_exe ./stockfish.master --c_chess_exe ./c-chess-cli --ordo_exe ./ordo --book_file_name ./noob_3moves.epd run96
```

Automatically converts all `.ckpt` found under `run96` to `.nnue` and runs games to find the best net. Games are played using `c-chess-cli` and nets are ranked using `ordo`.
This script runs in a loop, and will monitor the directory for new checkpoints. Can be run in parallel with the training, if idle cores are available.


# Thanks

* Sopel - for the amazing fast sparse data loader
* connormcmonigle - https://github.com/connormcmonigle/seer-nnue, and loss function advice.
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
* dkappe - Suggesting ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
