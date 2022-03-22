This file details instructions needed to setup the Late Fusion experiments

**Downloading Model Weights**

First, you must download the pretrained TDN weights from this link: https://drive.google.com/drive/folders/1N4EojVdHFfNbEU_4WIkT_26d_mSHimmo

Once downloaded, place it in the TDN folder

**Pre-extracting Deberta Features**

We use pre-extracted DeBERTa-V3 features to speed up the training process. To extract them, run:

```
conda activate PACS
python3 scripts/get_embeds.py
```

**Training**

To train on the PACS dataset use the following command:

```
python3 train.py -data_dir PATH_TO_DATASET_FOLDER -save_path PATH_TO_SAVE_MODEL --audio --video --text --image
```

You can include/exclude any of the four modalities of audio, video, text, and image when training

Similarly, you can train on the PACS-material dataset using the following command:

```
python3 train_material.py -data_dir PATH_TO_DATASET_FOLDER -save_path PATH_TO_SAVE_MODEL --audio --video --text --image
```
