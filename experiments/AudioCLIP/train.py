import torch
import numpy as np
import random
torch.random.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
from model import AudioCLIPFinetune
import argparse

from utils_extra import get_logger, load_model
from dataset import PACSImageAudioDataset
from torchvision import transforms
from transformations import RandomFlipQ, Tokenize
from transformers import AdamW
import torch.optim as optim
import os
from PIL import Image
import shutil
import utils.transforms as audio_transforms

parser = argparse.ArgumentParser(description='Extracting frames and audio')
parser.add_argument(
    '--data_dir',
    type=str,
)
parser.add_argument(
    '--save_path',
    default="logs/",
    type=str
)
parser.add_argument(
    '--run_num',
    default=1,
    type=int,
)
parser.add_argument(
    '--model_filename',
    default='AudioCLIP-Partial-Training.pt',
    type=str
)
parser.add_argument(
    '--batch_size',
    default=64,
    type=int,
)
parser.add_argument(
    '--lr',
    default=1e-4,
    type=float,
)
parser.add_argument(
    '--wd',
    default=1e-5,
    type=float,
)
parser.add_argument(
    '--num_epochs',
    default=40,
    type=int,
)
parser.add_argument(
    '--lr_steps',
    nargs='+',
    default=[20,30],
    type=int,
)
parser.add_argument(
    '--gamma',
    default=0.1,
    type=int,
)

args = parser.parse_args()

LOGGER_FILENAME = os.path.join(args.save_path, args.run_num, 'train.txt')
SAVE_PATH = os.path.join(args.save_dir, str(args.run_num))
os.makedirs(SAVE_PATH, exist_ok=True)
writer = tensorboard.SummaryWriter(log_dir=f"runs/{args.run_num}")

os.makedirs(f"logs/{args.run_num}", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME)

# Load the model
model = load_model("assets/ViT-B-16.pt", device)
logger.info("Model loaded")

audio_model = AudioCLIPFinetune(pretrained=f'assets/{args.model_filename}')
for param in audio_model.parameters():
    param.requires_grad = False

model.audio_model = audio_model.audio
model.text_projection = nn.Parameter(torch.empty(512, 512))
nn.init.normal_(model.text_projection, std=512 ** -0.5)
model.text_projection.data = model.text_projection.data.half()

model.audio_image_fuse = nn.Linear(1024+512,512)

for param in model.audio_image_fuse.parameters():
    param.data = param.data.half()

# Freeze all layers except the classifier
to_train = ["audio_image_fuse.weight", "audio_image_fuse.bias", "text_projection"] # 
for name, param in model.named_parameters():
    if name in to_train:
        param.requires_grad = True
        print(name)
    elif name.startswith("audio_image_fuse"):
        param.requires_grad = True
        print(name)
    else:
        param.requires_grad = False

model = model.to(device)

# Prepare the optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)

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
        transforms.Resize((224,224), interpolation=Image.BICUBIC),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


train_audio_transform = transforms.Compose([
        audio_transforms.ToTensor1D(),
        # audio_transforms.RandomScale(),
        audio_transforms.RandomCrop(out_len=44100*5), 
        audio_transforms.RandomPadding(out_len=44100*5),
        audio_transforms.RandomNoise(p=0.8),
        audio_transforms.RandomFlip(p=0.5),
])

train_q_transform = None
train_dataset = PACSImageAudioDataset(args.data_dir, "train_data", img_transform=train_img_transform, q_transform=train_q_transform, second_transform=train_second_transform, audio_transform=train_audio_transform, extra_imgs=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

val_img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

val_audio_transform = transforms.Compose([
        audio_transforms.ToTensor1D(),
        audio_transforms.RandomCrop(out_len=44100*5, train=False), 
        audio_transforms.RandomPadding(out_len=44100*5, train=False),
        ])

val_q_transform = None
val_dataset = PACSImageAudioDataset(args.data_dir, "val_data", img_transform=val_img_transform, q_transform=val_q_transform, test_mode=True, audio_transform=val_audio_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0

for epoch in range(args.num_epochs):
    SAVE = False
    train_total = 0
    train_correct = 0
    train_loss = 0
    iter_loss = 0 
    iter_total = 0
    iter_correct = 0

    model.train()
    for i, data in enumerate(train_dataloader):
        img1 = data["img1"].to(device)
        img2 = data["img2"].to(device)
        audio1 = data["audio1"].to(device)
        audio2 = data["audio2"].to(device)
        token = data["tokens"].to(device)
        
        objf1, objf2, textf1 = model(img1, audio1, img2, audio2, token)
        loss = loss_fn(textf1, objf2, objf1)
        cs1 = nn.CosineSimilarity()(textf1, objf1)
        cs2 = nn.CosineSimilarity()(textf1, objf2)
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
            logger.info("Epoch {}: Iter {}: Train loss: {}, Train acc: {}".format(epoch, i, iter_loss/iter_total*args.batch_size, iter_correct/iter_total))
            
            writer.add_scalar("train/loss", iter_loss/iter_total*args.batch_size, epoch*len(train_dataloader)+i)
            writer.add_scalar("train/acc", iter_correct/iter_total, epoch*len(train_dataloader)+i)

            iter_loss = 0
            iter_total = 0
            iter_correct = 0

    train_losses.append(train_loss / len(train_dataloader))
    train_accs.append(train_correct / train_total)

    # Validation
    model.eval()
    if epoch % 1 == 0:
        with torch.no_grad():
            val_loss = 0
            val_total = 0
            val_correct = 0

            for i, data in enumerate(val_dataloader):
                img1 = data["img1"].to(device)
                img2 = data["img2"].to(device)
                audio1 = data["audio1"].to(device)
                audio2 = data["audio2"].to(device)
                token = data["tokens"].to(device)

                objf1, objf2, textf1 = model(img1, audio1, img2, audio2, token)
                loss = loss_fn(textf1, objf2, objf1)
                cs1 = nn.CosineSimilarity()(textf1, objf1)
                cs2 = nn.CosineSimilarity()(textf1, objf2)
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

    # Save the model
    if SAVE:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_audioclip_{args.run_num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(SAVE_PATH))