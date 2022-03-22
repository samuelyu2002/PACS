## Requirements

The creators of UNITER provide a Docker image for easier reproduction. Please install the following:

- [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+),
- [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+),
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

The following steps were tested using a Titan Xp. Other GPUs from this generation and earlier will work, but I wasn't able to use the original UNITER docker images provided to run it on newer GPUs such as an RTX 3090.

## Dataset Setup

We must first preprocess some data:

1. Create a text_db by running

   ```bash
   docker run --ipc=host --rm -it \
           --mount src=$(pwd),dst=/src,type=bind \
           --mount src=PATH_TO_DATASET/txt_db,dst=/txt_db,type=bind \
           --mount src=PATH_TO_DATASET/json,dst=/ann,type=bind,readonly \
           -w /src chenrocks/uniter \
           python prepro_pacs.py 
   ```

2. There are two options. The easiest way to get the npz files is to download and extract the tar file from [here](https://drive.google.com/file/d/12_rOFFqki763AHYyIqQujHwrQOXoqWPs/view?usp=sharing). Alternatively, you can go into the docker file

   ```
   nvidia-docker run --gpus '"'device=0'"' --ipc=host --rm -it \
       --mount src=PATH_TO_DATASET/midframes,dst=/img,type=bind,readonly \
       --mount src=PATH_TO_DATASET,dst=/output,type=bind \
       -w /src chenrocks/butd-caffe:nlvr2
   ```

   Once in the docker, you must copy the generate_pacs_npz.py file into the /src/tools folder in the docker image, and run

   python tools/generate_pacs_npz.py

3. run

   ```
   export PATH_TO_STORAGE=YOUR_PATH_TO_STORAGE
   bash scripts/create_imgdb.sh PATH_TO_DATASET/pacs_npz $PATH_TO_STORAGE/img_db
   ```

   After doing this, your uniter storage folder should look like this:

   uniter_data
   ├── ann
   │   ├── test_data.json
   │   ├── train_data.json
   │   └── val_data.json
   ├── finetune
   │   └── pacs
   ├── img_db
   │   └── pacs_npz
   │       ├── feat_th0.2_max100_min10
   │       │   ├── data.mdb
   │       │   └── lock.mdb
   │       └── nbb_th0.2_max100_min10.json
   ├── pretrained
   │   └── uniter-large.pt
   └── txt_db
   ├── pacs_test.db
   │   ├── data.mdb
   │   ├── id2len.json
   │   ├── lock.mdb
   │   ├── meta.json
   │   └── txt2img.json
   ├── pacs_train.db
   │   ├── data.mdb
   │   ├── id2len.json
   │   ├── lock.mdb
   │   ├── meta.json
   │   └── txt2img.json
   └── pacs_val.db
   ├── data.mdb
   ├── id2len.json
   ├── lock.mdb
   ├── meta.json
   └── txt2img.json

   There may be missing folders, and in that case, create them to avoid errors.
   
   
4. Download the pretrained model using

```
bash scripts/download_pretrained.sh $PATH_TO_STORAGE
```

## Training

Launch the Docker container for running the experiments.

```bash
# docker image should be automatically pulled
source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
    $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
```

The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
Note that the source code is mounted into the container under `/src` instead
of built into the image so that user modification will be reflected without
re-building the image. (Data folders are mounted into the container separately
for flexibility on folder structures.)

To finetune PACS, use the following command:

```bash
   python train_pacs.py --config config/train-pacs-large.json
```

To run inference on PACS, run:

   ```bash
   # inference
   python inf_pacs.py --txt_db /txt/pacs_test.db/ --img_db /img// \
       --train_dir /storage/pacs-large/ --ckpt 8000 --output_dir . --fp16
   ```

## Acknowledgements

The code was adapted from [https://github.com/ChenRocks/UNITER](https://github.com/ChenRocks/UNITER), so we would like to thank Yen-Chun Chen and all other contributors of the repository.