import copy

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import random
random.seed(0)

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   get_ids_and_lens, pad_tensors, get_gather_index)


class PacsPairedDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, use_img_type=True, rand_box=False):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img
        self.lens = [2*tl + sum(self.img_db.name2nbb[img]
                                for img in txt2img[id_])
                     for tl, id_ in zip(txt_lens, self.ids)]

        self.use_img_type = use_img_type
        self.rand_box = rand_box

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        target = example['target']
        outs = []
        for i, img in enumerate(example['img_fname']):
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img)

            # half = num_bb // 2 + 1
            half = 1
            if self.rand_box:
                half = random.randint(half, num_bb)
                img_feat = img_feat[:half]
                img_pos_feat = img_pos_feat[:half]
                num_bb = half

            # text input
            input_ids = copy.deepcopy(example['input_ids'])

            input_ids = [self.txt_db.cls_] + input_ids + [self.txt_db.sep]
            attn_masks = [1] * (len(input_ids) + num_bb)
            input_ids = torch.tensor(input_ids)
            attn_masks = torch.tensor(attn_masks)

            if self.use_img_type:
                img_type_ids = torch.tensor([i+1]*num_bb)
            else:
                img_type_ids = None

            outs.append((input_ids, img_feat, img_pos_feat,
                         attn_masks, img_type_ids))
        if random.random() > 0.5 and self.rand_box:
            outs = [outs[1], outs[0]]
            target = 1-target
        return tuple(outs), target

def pacs_paired_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     img_type_ids) = map(list, unzip(concat(outs for outs, _ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.Tensor([t for _, t in inputs]).long()

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_type_ids': img_type_ids,
             'targets': targets}
    return batch


class PacsPairedEvalDataset(PacsPairedDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        outs, targets = super().__getitem__(i)
        return qid, outs, targets


def pacs_paired_eval_collate(inputs):
    qids, batch = [], []
    for id_, *tensors in inputs:
        qids.append(id_)
        batch.append(tensors)
    batch = pacs_paired_collate(batch)
    batch['qids'] = qids
    return batch

class PacsTripletDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, use_img_type=True, rand_box=False):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img
        self.lens = [tl + sum(self.img_db.name2nbb[img]
                              for img in txt2img[id_])
                     for tl, id_ in zip(txt_lens, self.ids)]

        self.use_img_type = use_img_type
        self.rand_box = rand_box

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        target = example['target']
        img_feats = []
        img_pos_feats = []
        num_bb = 0
        img_type_ids = []

        for i, img in enumerate(example['img_fname']):
            # print(i, img)
            feat, pos, nbb = self._get_img_feat(img)
            # half = nbb // 2 + 1
            # if self.rand_box:
            #     half = random.randint(half, nbb)
            #     feat = feat[:half]
            #     pos = pos[:half]
            #     nbb = half
            img_feats.append(feat)
            img_pos_feats.append(pos)
            num_bb += nbb
            if self.use_img_type:
                img_type_ids.append([i+1]*nbb)

        if random.random() > 0.5:
            img_feats = [img_feats[1], img_feats[0]]
            img_pos_feats = [img_pos_feats[1], img_pos_feats[0]]
            target = 1-target
            if self.use_img_type:
                temp = img_type_ids[0]
                img_type_ids = img_type_ids[1]
                img_type_ids.extend(temp)
        else:
            temp = img_type_ids[1]
            img_type_ids = img_type_ids[0]
            img_type_ids.extend(temp)
        
        img_feat = torch.cat(img_feats, dim=0)
        img_pos_feat = torch.cat(img_pos_feats, dim=0)
        if self.use_img_type:
            img_type_ids = torch.tensor(img_type_ids)
        else:
            img_type_ids = None

        # text input
        input_ids = copy.deepcopy(example['input_ids'])

        input_ids = [self.txt_db.cls_] + input_ids + [self.txt_db.sep]
        attn_masks = [1] * (len(input_ids) + num_bb)
        input_ids = torch.tensor(input_ids)
        attn_masks = torch.tensor(attn_masks)

        return (input_ids, img_feat, img_pos_feat, attn_masks,
                img_type_ids, target)


def pacs_triplet_collate(inputs):
    (input_ids, img_feats, img_pos_feats,
     attn_masks, img_type_ids, targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.Tensor(targets).long()

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_type_ids': img_type_ids,
             'targets': targets}
    return batch


class PacsTripletEvalDataset(PacsTripletDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        tensors = super().__getitem__(i)
        return (qid, *tensors)


def pacs_triplet_eval_collate(inputs):
    qids, batch = [], []
    for id_, *tensors in inputs:
        qids.append(id_)
        batch.append(tensors)
    batch = pacs_triplet_collate(batch)
    batch['qids'] = qids
    return batch
