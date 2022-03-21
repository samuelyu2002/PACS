import os
from torch.utils.data import Dataset
import json
import cv2
from PIL import Image, ImageFile
import clip
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

class PACSImageDataset(Dataset):
    def __init__(self, data_dir, split, img_transform=None, q_transform=None, second_transform=None, test_mode=False, extra_imgs=False):
        self.data_dir = data_dir
        self.split = split
        self.img_transform = img_transform
        self.second_transform = second_transform
        self.q_transform = q_transform
        self.test_mode = test_mode
        self.extra_imgs = extra_imgs
        self.data = self.load_data()
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.data[idx]

        img1 = Image.open(os.path.join(self.data_dir,"square_crop", sample['obj1'] + ".png"))
        img2 = Image.open(os.path.join(self.data_dir,"square_crop", sample['obj2'] + ".png"))

        if self.second_transform and random.random() < 0.1 and not self.test_mode:
            img1 = self.second_transform(img1)
            img2 = self.second_transform(img2)
        elif self.img_transform:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
        
        if sample["label"] == 0:
            target = {"obj1":sample["obj1"], "obj2":sample["obj2"], "question":sample["question"],"question_id":sample["question_id"], "tokens":sample["tokens"], "img1":img1, "img2":img2}
        else:
            target = {"obj1":sample["obj2"], "obj2":sample["obj1"], "question":sample["question"],"question_id":sample["question_id"], "tokens":sample["tokens"], "img1":img2, "img2":img1}

        return target

    def load_data(self):
        data_json = json.load(open(os.path.join(self.data_dir, 'json', self.split + '.json')))
        data = []
        for pair in data_json:
            v1, v2 = pair.split("_")

            if not (os.path.exists(os.path.join(self.data_dir,"square_crop", v1 + ".png"))):
                print(f"did not find image: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.data_dir,"square_crop", v2 + ".png"))):
                print(f"did not find image: {v2}")
                continue
            
            for q in data_json[pair]:
                sample = {"obj1":v1, "obj2":v2, "question":data_json[pair][q]["text"], "label":data_json[pair][q]["label"], "question_id":q, "tokens":clip.tokenize(data_json[pair][q]["text"])[0]}

                data.append(sample)
            
        return data