import torch
import numpy as np
import torch.nn as nn
from utils import load_model
from torchvision import transforms
from PIL import Image
import json
import clip
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser(description='Extracting frames and audio')
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
        default="test",
        type=str,
        help='which split to predict'
    )

args = parser.parse_args()

MODEL_PATH = args.model_path
PRE_LOAD = "ViT-B-16.pt"
DATA_DIR = args.data_dir
SPLIT = args.split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(PRE_LOAD, device)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint)


img_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

img_transform2 = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

similarities = defaultdict(dict)


test_data = json.load(open(f"{DATA_DIR}/json/{SPLIT}.json", 'r'))

with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    correct0 = 0
    total0 = 0
    correct1 = 0
    total1 = 0
    for pair in test_data:
        obj1, obj2 = pair.split("\\\\")
        img1 = Image.open(f"{DATA_DIR}/square_crop/{obj1}.png")
        img2 = Image.open(f"{DATA_DIR}/square_crop/{obj2}.png")

        img1_1 = Image.open(f"{DATA_DIR}/square_crop/{obj1}.png")
        img2_1 = Image.open(f"{DATA_DIR}/square_crop/{obj2}.png")

        img1 = img_transform(img1)
        img1_2 = img_transform2(img1_1).reshape(1,3,224,224)

        img1 = torch.cat([img1, img1_2], dim=0).to(device)

        img2 = img_transform(img2)
        img2_2 = img_transform2(img2_1).reshape(1, 3, 224, 224)
        img2 = torch.cat([img2, img2_2], dim=0).to(device)

        for q in test_data[pair]:
            text = test_data[pair][q]["text"]
            tokens = clip.tokenize(text).to(device)

            imgf1, imgf2, textf1 = model(img1, img2, tokens)
            cs1 = nn.CosineSimilarity()(textf1, imgf1)
            cs2 = nn.CosineSimilarity()(textf1, imgf2)

            cs1 = cs1.detach().cpu().numpy()
            cs2 = cs2.detach().cpu().numpy()

            similarities[pair][q] = [cs1.tolist(), cs2.tolist()]

            cs1 = np.sum(cs1) + 2*cs1[-1]
            cs2 = np.sum(cs2) + 2*cs2[-1]

            if test_data[pair][q]["label"] == 0:
                if cs1 > cs2:
                    correct0 += 1
                    correct += 1
                total0 += 1
                
            elif test_data[pair][q]["label"] == 1:
                if cs1 < cs2:
                    correct1 += 1
                    correct += 1
                total1 += 1
            total += 1

print(correct, total, correct/total)
print(correct0, total0, correct0/total0)
print(correct1, total1, correct1/total1)
        

json.dump(dict(similarities), open(os.path.join(args.save_dir, f"results_{SPLIT}.json"), 'w'))