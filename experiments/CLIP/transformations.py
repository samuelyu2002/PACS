import random
import torchvision.transforms.functional as F
import clip

class RandomFlipQ(object):
    """Flip the objects in the sample such that the correct answer is the opposite with probability p.

    Args:
        p (float): Probability of flipping the sample.
    """

    def __init__(self, p=0.5):
        assert 0 <= p <= 1, "Probability must be between 0 and 1"
        self.p = p

    def __call__(self, sample):
        obj1 = sample["obj1"]
        obj2 = sample["obj2"]
        img1 = sample["img1"]
        img2 = sample["img2"]
        label = sample["label"]

        if random.random() < self.p:
            # print("hi")
            sample["obj1"] = obj2
            sample["obj2"] = obj1
            sample["img1"] = img2
            sample["img2"] = img1
            sample["label"] = 1 - label

        return sample

class Tokenize(object):
    """Tokenize the question.
    """

    def __call__(self, sample):
        question = sample["question"]
        tokens = clip.tokenize(question)
        sample["tokens"] = tokens[0]

        return sample
