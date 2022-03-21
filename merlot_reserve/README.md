This file details instructions needed to setup the merlot reserve experiments. Unfortunately, finetuning the model requires TPUs.

First, set up a conda environment locally:


```bash
conda create --name mreserve python=3.8 && conda activate mreserve
conda install -y python=3.8 tqdm numpy pyyaml scipy ipython cython typing h5py pandas matplotlib

# Install jax
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# If doing this on TPUs instead of locally...
# pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# This is needed sometimes https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp
pip uninstall numpy
pip install numpy==1.19.5

pip install -r requirements.txt
```



**Setting up the Dataset**

First, you must edit line 38 in the finetune/prep_data.py file, and enter the directory where the entire dataset is stored. Once you do that, run:

sh finetune/prep_data.sh 

to create the tfrecord files. 

Once the tfrecord files have been created, you should upload the tfrecord files to google cloud storage.

**Training**

As mentioned previously, training requires TPUs, so first, you must follow these instructions:


First, create a google cloud machine (which has access to the needed API to create your Cloud TPU VM)
* Create machine configuration with `Compute-optimized` and `c2-standard-4` in your desired region
* Boot disk `Debian GNU/Linux 10 Buster + TF 2-5-0`
* Add SSH key and static external IP address. Network=main. Then add to `~/.ssh/config` on the local machine
* Last under "Cloud API access scopes" you must Allow Full Access to All Cloud APIs

* You might need to do something weird about firewall allowing ssh [https://cloud.google.com/tpu/docs/users-guide-tpu-vm](see the users guide)
    * `gcloud compute firewall-rules create --network=default allow-ssh --allow=tcp:22` 
* Log in and do `gcloud config set compute/zone ${MYZONE}` to the zone you want
* Generate new ssh keys on the cloud machine -- just do it by sshing into a TPU
```
gcloud alpha compute tpus tpu-vm create TEST --zone europe-west4-a
gcloud alpha compute tpus tpu-vm ssh TEST --zone europe-west4-a --dry-run
```

If those steps work, you can then ssh into the TPU:

gcloud alpha compute tpus tpu-vm ssh TEST --zone=europe-west4-a

Now, the next steps are all done locally on the TPU

Then, you must copy the code in this folder onto the TPU - the recommended way to do so is using google cloud storage, eg:

```
gsutil cp gs://your_cloud_bucket/tarred_file tarred_file

tar -xvf tarred_file

cd merlot_reserve
```

Then, run the following commands:

```
pip install "jax[tpu]>=0.2.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu fabric dataclasses optax flax tqdm cloudpickle smart_open[gcs] func_timeout aioredis==1.3.1 transformers wandb pandas
pip uninstall tensorflow
pip install tensorflow==2.6.0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368
gsutil cp gs://merlotreserve/ckpts/large_resadapt large_resadapt
gsutil cp gs://merlotreserve/ckpts/base_resadapt base_resadapt
```

Now, before training, you must edit lines 109, 142, 157, and 332 in the finetune/pacs/pacs_finetune.py file to include your google cloud storage bucket

Similarly, edit lines 109, 142, 157, and 362 in the finetune/pacs/pacs_finetune_audio.py file to include your cloud storage bucket

Then, you can run 

```
cd finetune/pacs
python3 pacs_finetune.py ../../pretrain/configs/base.yaml ../../base_resadapt -lr=5e-6 -ne=40 -wd=1e-7 -output_grid_h=18 -output_grid_w=32
python3 pacs_finetune_audio.py ../../pretrain/configs/large.yaml ../../large_resadapt -lr=5e-6 -ne=40 -wd=1e-7 -output_grid_h=18 -output_grid_w=32
```

To train versions with and without audio, and base and large models.
