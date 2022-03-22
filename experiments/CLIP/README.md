This file details instructions needed to setup CLIP experiments

## Downloading Model Weights

First, you must download the pretrained CLIP weights

CLIP:

```
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

Also, download bpe_simple_vocab_16e6.txt.gz from https://github.com/AndreyGuzhov/AudioCLIP/releases

Once downloaded, put the files in the CLIP folder

## Training

To train on the PACS dataset use the following command:

```
conda activate PACS
python3 train.py -data_dir PATH_TO_DATASET_FOLDER -save_path PATH_TO_SAVE_MODEL
```

## Prediction

Once a model has been trained, we can also generate predicted outputs on the test set:

```
python3 predict.py -model_path PATH_TO_MODEL_WEIGHTS -save_dir PATH_TO_SAVE_RESULTS -split test
```

## Acknowledgements

The code was adapted from [https://github.com/openai/CLIP](https://github.com/openai/CLIP), so we would like to thank the contributors of the repository.