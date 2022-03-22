import argparse
import json
import pickle
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

def process_pacs(data, db, tokenizer, split, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    GLOBAL_COUNT = 0
    for pair in tqdm(data, desc='processing PACS'):
        img1, img2 = pair.split("_")
        img_fname = (f'pacs_{img1}.npz', f'pacs_{img2}.npz')
        for q in data[pair]:
            example = {}
            GLOBAL_COUNT += 1
            # if GLOBAL_COUNT%1000 == 0:
            #     print(GLOBAL_COUNT)
            id_ = split + "_" + str(GLOBAL_COUNT)
            input_ids = tokenizer(data[pair][q]["text"])
            target = data[pair][q]['label']
            txt2img[id_] = img_fname
            id2len[id_] = len(input_ids)
            example['input_ids'] = input_ids
            example['target'] = target
            example["img_fname"] = img_fname
            example["identifier"] = id_
            example["question"] = data[pair][q]["text"]
            db[id_] = example
            data[pair][q]["id"] = id_
    return id2len, txt2img            

def main(opts):
    for split in ['train', 'val', 'test']:
        output_opt = f"/txt_db/pacs_{split}.db"
        if not exists(output_opt):
            os.makedirs(output_opt)
        else:
            raise ValueError('Found existing DB. Please explicitly remove '
                            'for re-processing')
        meta = vars(opts)
        meta['tokenizer'] = opts.toker
        toker = BertTokenizer.from_pretrained(
            opts.toker, do_lower_case='uncased' in opts.toker)
        tokenizer = bert_tokenize(toker)
        meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
        meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
        meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
        meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
        meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                        len(toker.vocab))
        with open(f'{output_opt}/meta.json', 'w') as f:
            json.dump(vars(opts), f, indent=4)

        open_db = curry(open_lmdb, output_opt, readonly=False)
        output_field_name = ['id2len', 'txt2img']
        with open_db() as db:
            data = json.load(open(f"/ann/{split}_data.json", 'r'))
            jsons = process_pacs(data, db, tokenizer, split)
            json.dump(data, open(f"/src/{split}_data_uniter.json", 'w'))

        for dump, name in zip(jsons, output_field_name):
            with open(f'{output_opt}/{name}.json', 'w') as f:
                json.dump(dump, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    
    main(args)
