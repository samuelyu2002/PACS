import sys
sys.path.append("ast/src/models")

import torch
torch.manual_seed(6361)
import random
random.seed(735745745)
import numpy as np
np.random.seed(52534)
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, ViTFeatureExtractor, ViTModel, AdamW, DebertaV2Tokenizer, AlbertTokenizer, AlbertModel
from PIL import Image
from ast_models import ASTModel
from model import Identity
from TDN.ops.models import TSN
import torchvision
from torchvision import transforms
from TDN.ops.transforms import *
from transformations import Tokenize, RandomFlipQ, PreTokenize
from dataset import PACSFusionDataset
import torch.optim as optim
from model import FusionModel
from utils import get_logger
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import timm
import argparse

#######################################
# Hyperparameters and Other Constants #
#######################################

parser = argparse.ArgumentParser(description='train late fusion')
parser.add_argument(
        '-data_dir',
        dest='data_dir',
        default="../dataset/",
        type=str,
        help='Directory containing PACS data'
    )
parser.add_argument(
        '-save_path',
        dest='save_dir',
        default="logs/",
        type=str,
        help='Directory containing PACS data'
    )
parser.add_argument(
        '-run_num',
        dest='run_num',
        default=1,
        type=int,
        help='Directory containing PACS data'
    )
parser.add_argument('--audio', action='store_true')
parser.add_argument('--video', action='store_true')
parser.add_argument('--text', action='store_true')
parser.add_argument('--image', action='store_true')

args = parser.parse_args()

devices = ["cuda:0"]
num = args.run_num

os.makedirs(f"logs/{num}", exist_ok=True)

# TODO: Change to parser args
LOGGER_FILENAME = f"logs/{num}/log.txt"
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.00005
BATCH_SIZE = 32
WARMUP_PROPORTION = -1
TOTAL_EPOCHS = 40
SAVE_PATH = os.path.join(args.save_dir, str(num))
MID_DIM = 768
DROPOUT = 0.2
LOG_INTERVAL = 10
LR_DROPS = [20, 30]
NUM_WORKERS = 4
USE_VID = args.video
USE_AUDIO = args.audio
USE_TEXT = args.text
USE_IMAGE = args.image
assert USE_AUDIO or USE_TEXT or USE_IMAGE or USE_VID, "Must use at least one modality"
VID_FEATS = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME)

writer = SummaryWriter(log_dir=f"runs/{num}")

#######################
# Load all the models #
#######################

# Load the text model
if USE_TEXT:
    tokenizer = "deberta"
    if tokenizer == "albert":
        TEXT_FEATS=768
    elif tokenizer == "deberta":
        TEXT_FEATS=1024
else:
    tokenizer = "deberta"
    TEXT_FEATS = 0

# Load the image model and feature extractor
if USE_IMAGE:
    image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    norm_mean = (0.5,0.5,0.5)
    norm_std = (0.5,0.5,0.5)
    IMAGE_FEATS=768
    for param in image_model.parameters():
        param.requires_grad = False
    logger.info("Image model loaded")
else:
    IMAGE_FEATS=0
    image_model = None
    norm_mean = (0.5,0.5,0.5)
    norm_std = (0.5,0.5,0.5)

# Load the audio model
if USE_AUDIO:
    input_tdim = 1024
    audio_model = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    checkpoint_path = "../pretrained_models/audioset_10_10_0.4593.pth"
    checkpoint_audio = torch.load(checkpoint_path, map_location=device) 
    checkpoint_audio = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint_audio.items())}
    audio_model.load_state_dict(checkpoint_audio)
    audio_model.mlp_head = Identity()
    for param in audio_model.parameters():
        param.requires_grad = False
    logger.info("Audio model loaded")
else:
    audio_model = None

# Load the video model
if USE_VID:
    checkpoint_video = torch.load("TDN/best8.pth.tar", map_location=device)
    video_model = TSN(174, 8, "RGB",
                base_model="resnet101",
                new_length=1, 
                consensus_type='avg',
                img_feature_dim=256,
                pretrain="imagenet"
                )
    checkpoint_video = checkpoint_video['state_dict']
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint_video.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)
    video_model.load_state_dict(base_dict)
    this_arch = 'resnet101'
    input_size = video_model.scale_size
    cropping = transforms.Compose([
                GroupScale(video_model.scale_size),
                GroupCenterCrop(input_size),
            ])
    video_model.new_fc = Identity()
    for param in video_model.parameters():
        param.requires_grad = False
    logger.info("Video model loaded")
else:
    video_model = None

####################
# Load the Dataset #
####################

train_q_transform = transforms.Compose([PreTokenize(tokenizer), RandomFlipQ()])
train_img_transform = transforms.Compose([
        # transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((224,224), (0.85, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

if USE_VID:
    train_augmentation = video_model.get_augmentation(
            flip=False)
    vid_normalize = GroupNormalize(video_model.input_mean, video_model.input_std)
    train_v_transform = transforms.Compose([train_augmentation,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                vid_normalize])
else:
    train_v_transform = None

train_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 48, 'timem': 144, 'mode': 'train', 'dataset': 'audioset', 'mean': -4.2677393, 'std': 4.5689974, 'noise': True}

train_dataset = PACSFusionDataset(args.data_dir, "train_data", train_audio_conf, img_transform=train_img_transform, q_transform=train_q_transform, v_transform=train_v_transform, use_vid=USE_VID, use_audio=USE_AUDIO)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

val_q_transform = transforms.Compose([PreTokenize(tokenizer)])
val_img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

if USE_VID:
    cropping = torchvision.transforms.Compose([
                GroupScale(video_model.scale_size),
                GroupCenterCrop(video_model.scale_size),
            ])
    vid_normalize = GroupNormalize(video_model.input_mean, video_model.input_std)
    val_v_transform = transforms.Compose([cropping,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                vid_normalize])
else:
    val_v_transform = None

val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0,'mode': 'test', 'dataset': 'audioset', 'mean': -4.2677393, 'std': 4.5689974, 'noise' : False}

val_dataset = PACSFusionDataset(args.data_dir, "train_data", val_audio_conf, img_transform=val_img_transform, q_transform=val_q_transform, v_transform=val_v_transform, test_mode=True, random_shift=False, use_vid=USE_VID, use_audio=USE_AUDIO)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

logger.info("Dataset loaded")

########################
# Load the Final Model #
########################

model = FusionModel(image_model, audio_model=audio_model, video_model=video_model, mid_dim=MID_DIM, dropout=DROPOUT, vid_feats=VID_FEATS, text_feats=TEXT_FEATS, img_feats=IMAGE_FEATS, text_model=tokenizer)
model = torch.nn.DataParallel(model, device_ids=devices)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DROPS, gamma=0.1)
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(device)

logger.info("Model loaded")

total_iters = 0
best_val_acc = 0

for epoch in range(TOTAL_EPOCHS):

    ############
    # Training #
    ############

    train_total = 0
    train_correct = 0
    train_loss = 0
    model.train()
    iters = len(train_dataloader)
    prev_total = 0
    prev_loss = 0
    prev_correct = 0
    SAVE = False
    for i, data in enumerate(train_dataloader):
        total_iters += 1
        img1 = data["img1"]
        img2 = data["img2"]
        vid1 = data["vid1"]
        vid2 = data["vid2"]
        audio1 = data["audio1"]
        audio2 = data["audio2"]
        embeds = data["embeddings"]
        label = data["label"]
        
        img1 = img1.to(device)
        img2 = img2.to(device)
        vid1 = vid1.to(device)
        vid2 = vid2.to(device)
        audio1 = audio1.to(device)
        audio2 = audio2.to(device)
        embeds = embeds.to(device)
        label = label.float().to(device)
    
        oup = model(embeds, img1, vid1, audio1, img2, vid2, audio2)
        pred = torch.sigmoid(oup)
        oup = oup.squeeze()
        loss = loss_fn(oup.squeeze(), label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (pred.squeeze() > 0.5)
        train_total += label.size(0)
        train_correct += (preds == label).sum().item()
        train_loss += loss.item()

        prev_total += label.size(0)
        prev_loss += loss.item()
        prev_correct += (preds == label).sum().item()

        if i % LOG_INTERVAL == LOG_INTERVAL-1:
            logger.info("Epoch {}: Iter: {} Train loss: {}, Train acc: {}".format(epoch, i, prev_loss/prev_total*BATCH_SIZE, prev_correct/prev_total))

            writer.add_scalar('Train/Loss', prev_loss/prev_total*BATCH_SIZE, total_iters)
            writer.add_scalar('Train/Acc', prev_correct/prev_total, total_iters)

            prev_total = 0
            prev_loss = 0
            prev_correct = 0
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_total = 0
        val_correct = 0

        for i, data in enumerate(val_dataloader):
            img1 = data["img1"]
            img2 = data["img2"]
            vid1 = data["vid1"]
            vid2 = data["vid2"]
            audio1 = data["audio1"]
            audio2 = data["audio2"]
            embeds = data["embeddings"]
            label = data["label"]

            img1 = img1.to(device)
            img2 = img2.to(device)
            vid1 = vid1.to(device)
            vid2 = vid2.to(device)
            audio1 = audio1.to(device)
            audio2 = audio2.to(device)
            embeds = embeds.to(device)
            label = label.float().to(device)

            oup = model(embeds, img1, vid1, audio1, img2, vid2, audio2)
            pred = torch.sigmoid(oup)
            oup = oup.squeeze()

            loss = loss_fn(oup.squeeze(), label)

            preds = (pred.squeeze() > 0.5)
            val_total += label.size(0)
            val_correct += (preds == label).sum().item()
            val_loss += loss.item()

        logger.info("Epoch {}: Val loss: {}, Val acc: {}".format(epoch, val_loss / len(val_dataloader), val_correct/val_total))

        writer.add_scalar('Val/Loss', val_loss / len(val_dataloader), epoch)
        writer.add_scalar('Val/Acc', val_correct/val_total, epoch)

        if val_correct/val_total > best_val_acc:
            best_val_acc = val_correct/val_total
            SAVE =True

    # Save the model
    if SAVE:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_{num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(SAVE_PATH))
    
    scheduler.step()