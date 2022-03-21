# PACS

## Dataset Download

Before downloading the dataset, it is recommended to create an Anaconda environment:

```
conda create --name PACS python=3.8.11
conda activate PACS
pip install -r requirements.txt
```

Then, install the correct version of PyTorch, based on your cuda version [here](https://pytorch.org/get-started/locally/). For example:

```
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

To download the dataset, run:

```
cd dataset/scripts
python3 download.py -data_dir PATH_TO_DATA_STORAGE_HERE
python3 preprocess.py -data_dir PATH_TO_DATA_STORAGE_HERE
```

## Baseline Models

To run baseline models, visit the corresponding folders in our repository. 