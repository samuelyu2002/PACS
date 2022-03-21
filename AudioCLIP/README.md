This file details instructions needed to setup AudioCLIP experiments

**Downloading Model Weights**

First, you must download the pretrained CLIP and AudioCLIP weights

CLIP:

```
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

AudioCLIP: download AudioCLIP-Partial-Training.py and bpe_simple_vocab_16e6.txt.gz from https://github.com/AndreyGuzhov/AudioCLIP/releases

Once downloaded, put the models into the assets folder

**Training**

There are two models that can be trained, the first is on the PACS dataset:

```
conda activate PACS
python3 train.py -data_dir PATH_TO_DATASET_FOLDER -save_path PATH_TO_SAVE_MODEL
```

The second is for the material classification dataset:

```
python3 train_classify.py -data_dir PATH_TO_DATASET_FOLDER -save_path PATH_TO_SAVE_MODEL
```

**Prediction**

Once a model has been trained, we can also generate predicted outputs on the test set:

```
python3 predict.py -model_path PATH_TO_MODEL_WEIGHTS -save_dir PATH_TO_SAVE_RESULTS -split test
```
