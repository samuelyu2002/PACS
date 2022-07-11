import torch
import numpy as np
import torch.nn as nn
from utils_extra import load_model
from torchvision import transforms
from PIL import Image
import json
import clip
from collections import defaultdict
from model import AudioCLIPFinetune
import utils.transforms as audio_transforms
import librosa
import os
import argparse


parser = argparse.ArgumentParser(description='predict on audioclip')
parser.add_argument(
        '-model_path',
        dest='model_path',
        type=str,
        help='Model path'
    )
parser.add_argument(
        '-save_dir',
        dest='save_dir',
        default="results/",
        type=str,
        help='Directory containing PACS data'
    )

parser.add_argument(
        '-split',
        dest='split',
        default="test_data",
        type=str,
        help='which split to predict'
    )

parser.add_argument(
        '-data_dir',
        dest='data_dir',
        type=str,
        help='which split to predict'
    )

args = parser.parse_args()

PRELOAD = "assets/AudioCLIP-Partial-Training.pt"
MODEL_PATH = args.model_path

PRE_LOAD = "assets/ViT-B-16.pt"
DATA_DIR = args.data_dir
SPLIT = args.split

audio_model = AudioCLIPFinetune(pretrained=f'{PRELOAD}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(PRE_LOAD, device)
model.audio_model = audio_model.audio
model.text_projection = nn.Parameter(torch.empty(512, 512))
model.audio_image_fuse = nn.Linear(1024+512,512)

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint)
model.text_projection.data = model.text_projection.data.half()
for param in model.audio_image_fuse.parameters():
    param.data = param.data.half()
model = model.to(device)

img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

audio_transform = transforms.Compose([
        audio_transforms.ToTensor1D(),
        audio_transforms.RandomCrop(out_len=44100*5, train=False), 
        audio_transforms.RandomPadding(out_len=44100*5, train=False),
        ])

similarities = defaultdict(dict)

test_data = json.load(open(f"{DATA_DIR}/json/{SPLIT}.json", 'r'))

with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for pair in test_data:
        obj1, obj2 = pair.split("_")
        img1 = Image.open(f"{DATA_DIR}/square_crop/{obj1}.png")
        img2 = Image.open(f"{DATA_DIR}/square_crop/{obj2}.png")

        img1 = img_transform(img1).to(device)
        img2 = img_transform(img2).to(device)

        img1 = img1.reshape(1,3,224,224)
        img2 = img2.reshape(1,3,224,224)

        audio1, _ = librosa.load(os.path.join(DATA_DIR, "audio44100", obj1 + ".wav"), sr=44100, mono=True, dtype=np.float32)
        audio2, _ = librosa.load(os.path.join(DATA_DIR, "audio44100", obj2 + ".wav"), sr=44100, mono=True, dtype=np.float32)
        if audio1.ndim == 1:
            audio1 = audio1[:, np.newaxis]
            audio2 = audio2[:, np.newaxis]
        
        audio1 = audio1.T
        audio2 = audio2.T

        audio1 = audio_transform(audio1)
        audio2 = audio_transform(audio2)

        audio1 = torch.stack([audio1]*img1.shape[0]).to(device)
        audio2 = torch.stack([audio2]*img2.shape[0]).to(device)

        for q in test_data[pair]:
            text = test_data[pair][q]["text"]
            tokens = clip.tokenize(text).to(device)

            imgf1, imgf2, textf1 = model(img1, audio1, img2, audio2, tokens)
            cs1 = nn.CosineSimilarity()(textf1, imgf1)
            cs2 = nn.CosineSimilarity()(textf1, imgf2)

            cs1 = cs1.detach().cpu().numpy()
            cs2 = cs2.detach().cpu().numpy()

            similarities[pair][q] = [cs1.tolist()[0], cs2.tolist()[0]]

            if test_data[pair][q]["label"] == 0:
                if cs1 > cs2:
                    correct += 1 
            elif test_data[pair][q]["label"] == 1:
                if cs1 < cs2:
                    correct += 1

            total += 1

print(correct, total, correct/total)
        
json.dump(dict(similarities), open(os.path.join(args.save_dir, f"preds_{SPLIT}.json"), 'w'))