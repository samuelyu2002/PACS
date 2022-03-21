import random
import torchvision.transforms.functional as F
import clip
import torch
import json
import numpy as np
import os

class RandomFlipQ(object):
    """Flip the objects in the sample such that the correct answer is the opposite with probability p.

    Args:
        p (float): Probability of flipping the sample.
    """

    def __init__(self, p=0.5):
        assert 0 <= p <= 1, "Probability must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        if 'obj1' in sample:
            obj1 = sample["obj1"]
            obj2 = sample["obj2"]
        if 'img1' in sample:
            img1 = sample["img1"]
            img2 = sample["img2"]
        if "vid1" in sample:
            vid1 = sample["vid1"]
            vid2 = sample["vid2"]
        if "audio1" in sample:
            audio1 = sample["audio1"]
            audio2 = sample["audio2"]
        label = sample["label"]

        if random.random() < self.p:
            # print("hi")
            if 'obj1' in sample:
                sample["obj1"] = obj2
                sample["obj2"] = obj1
            if 'img1' in sample:
                sample["img1"] = img2
                sample["img2"] = img1
            if "vid1" in sample:
                sample["vid1"] = vid2
                sample["vid2"] = vid1
            if "audio1" in sample:
                sample["audio1"] = audio2
                sample["audio2"] = audio1
            sample["label"] = 1 - label

        return sample

class ReplaceSynonyms(object):
    """Replace common words in the question with synonyms.

    Args:
        p (float): Probability of replacing a word with a synonym.
    """

    def __init__(self, p=0.1):
        assert 0 <= p <= 1, "Probability must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        raise NotImplementedError
        return sample

class FlipSentiment(object):
    """Flip the sentiment of the question.

    Args: 
        p (float): Probability of flipping the sentiment of the question.
    """

    def __init__(self, p=0.5):
        assert 0 <= p <= 1, "Probability must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        raise NotImplementedError
        return sample

class Tokenize(object):
    """Tokenize the question.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sample, max_seq_len=256):
        question = sample["question"]
        tokens = self.tokenizer.tokenize(question)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1]*len(input_ids)

        paddings = max_seq_len - len(tokens)
        input_ids = input_ids + [0]*paddings
        attention_mask = attention_mask + [0]*paddings

        sample["tokens"] = {
            'input_ids': torch.tensor(input_ids, dtype=torch.int),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int)
            }
        return sample

class TokenizedMaterials(object):
    """Tokenize the question.
    """

    def __init__(self, data_root="../dataset/json", tokenizer="deberta"):
        self.data = json.load(open(os.path.join(data_root, f'mat_{tokenizer}_embeds.json')))

    def __call__(self, sample):

        target = str(sample["target"])
        num = random.randint(0, len(self.data[target])-1)
        embeds = self.data[target][num]
        embeds = np.array(embeds).astype(np.float32)
        embeds = torch.tensor(embeds)
        sample["embeddings"] = embeds
        return sample

class PreTokenize(object):
    def __init__(self, tokenizer, data_root="../dataset/json/"):
        self.tokenizer = tokenizer
        self.data = json.load(open(os.path.join(data_root, f"{tokenizer}_embeddings.json"), 'r'))

    def __call__(self, sample):
        embeddings = self.data[sample["id"]]
        embeddings = np.array(embeddings).astype(np.float32)
        embeddings = torch.tensor(embeddings)
        sample["embeddings"] = embeddings
        return sample
