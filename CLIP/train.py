import torch
import numpy as np
import random
import torch.nn as nn
torch.random.manual_seed(123421)
np.random.seed(12512)
random.seed(62341)
import torch.utils.tensorboard as tensorboard
from utils import get_logger, load_model
from dataset import PACSImageDataset
from torchvision import transforms
from transformations import RandomFlipQ, Tokenize
from transformers import AdamW
import torch.optim as optim
import os
from PIL import Image
import shutil
import argparse

parser = argparse.ArgumentParser(description='Extracting frames and audio')
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

args = parser.parse_args()
 

# TODO: make params into parser arguments
num = args.run_num
LOGGER_FILENAME = f"logs/{num}/train.txt"
BATCH_SIZE = 64
MODEL_PATH = "ViT-B-16.pt"
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001
WARMUP_PROPORTION = -1
TOTAL_EPOCHS = 40
DATA_DIR = args.data_dir
INPUT_SIZE = 224
LR_DROPS = [20, 30]
SAVE_PATH = os.path.join(args.save_dir, str(num))
os.makedirs(SAVE_PATH, exist_ok=True)
writer = tensorboard.SummaryWriter(log_dir=f"runs/{num}")

os.makedirs(f"logs/{num}", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME)

# Load the model
model = load_model(MODEL_PATH, device)
logger.info("Model loaded from {}".format(MODEL_PATH))

# Freeze all layers except the classifier
for name, param in model.named_parameters():
    if "text_projection" in name:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False

# Prepare the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DROPS, gamma=0.2)

# Loss function
loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1)

# Load the Dataset
train_img_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224), (0.85, 1.0), ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
train_second_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
train_q_transform = None
train_dataset = PACSImageDataset(DATA_DIR, "train_data", img_transform=train_img_transform, q_transform=train_q_transform, second_transform=train_second_transform, extra_imgs=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

val_img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

val_q_transform = None
val_dataset = PACSImageDataset(DATA_DIR, "val_data", img_transform=val_img_transform, q_transform=val_q_transform, test_mode=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0



for epoch in range(TOTAL_EPOCHS):
    SAVE = False
    train_total = 0
    train_correct = 0
    train_loss = 0
    iter_loss = 0 
    iter_total = 0
    iter_correct = 0
    for i, data in enumerate(train_dataloader):
        img1 = data["img1"]
        img2 = data["img2"]
        token = data["tokens"]

        img1 = img1.to(device)
        img2 = img2.to(device)
        token = token.to(device)

        imgf1, imgf2, textf1 = model(img1, img2, token)
        loss = loss_fn(textf1, imgf2, imgf1)
        cs1 = nn.CosineSimilarity()(textf1, imgf1)
        cs2 = nn.CosineSimilarity()(textf1, imgf2)
        correct = (cs1 > cs2).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total += token.size(0)
        train_correct += correct
        train_loss += loss.item()

        iter_total += token.size(0)
        iter_correct += correct
        iter_loss += loss.item()

        if i % 40 == 39:
            logger.info("Epoch {}: Iter {}: Train loss: {}, Train acc: {}".format(epoch, i, iter_loss/iter_total*BATCH_SIZE, iter_correct/iter_total))
            
            writer.add_scalar("train/loss", iter_loss/iter_total*BATCH_SIZE, epoch*len(train_dataloader)+i)
            writer.add_scalar("train/acc", iter_correct/iter_total, epoch*len(train_dataloader)+i)

            iter_loss = 0
            iter_total = 0
            iter_correct = 0

    train_losses.append(train_loss / len(train_dataloader))
    train_accs.append(train_correct / train_total)

    # Validation
    if epoch % 1 == 0:
        with torch.no_grad():
            val_loss = 0
            val_total = 0
            val_correct = 0

            for i, data in enumerate(val_dataloader):
                img1 = data["img1"]
                img2 = data["img2"]
                token = data["tokens"]

                img1 = img1.to(device)
                img2 = img2.to(device)
                token = token.to(device)

                imgf1, imgf2, textf1 = model(img1, img2, token)
                loss = loss_fn(textf1, imgf2, imgf1)

                cs1 = nn.CosineSimilarity()(textf1, imgf1)
                cs2 = nn.CosineSimilarity()(textf1, imgf2)
                correct = (cs1 > cs2).sum().item()

                val_total += token.size(0)
                val_correct += correct

                val_loss += loss.item()
            
            val_losses.append(val_loss / len(val_dataloader))
            val_accs.append(val_correct / val_total)

            logger.info("Epoch {}: Val loss: {}, Val acc: {}".format(epoch, val_loss / len(val_dataloader), val_correct/val_total))

            writer.add_scalar("val/loss", val_loss / len(val_dataloader), epoch)
            writer.add_scalar("val/acc", val_correct/val_total, epoch)

            if val_correct/val_total >= best_val_acc:
                best_val_acc = val_correct/val_total
                SAVE = True

    if SAVE:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_{num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(SAVE_PATH))