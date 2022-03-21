"""
Convert PACS into tfrecords
"""
import sys

sys.path.append('../../')
import argparse
import hashlib
import io
import json
import os
import random
import numpy as np
from tempfile import TemporaryDirectory
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from google.cloud import storage
from sacremoses import MosesDetokenizer
import regex as re
from tqdm import tqdm
import pandas as pd
from finetune.common_data_utils import *
from collections import defaultdict
import colorsys
import hashlib
import tempfile
import subprocess
from scipy.io import wavfile
from mreserve.preprocess import make_spectrogram2, invert_spectrogram
from mreserve.lowercase_encoder import START

parser = create_base_parser()

args = parser.parse_args()

DATA_DIR = 'PUT_YOUR_DIRECTORY_HERE'
os.makedirs(f"{DATA_DIR}/mreserve_data", exist_ok=True)

random.seed(123456)

out_fn = os.path.join(DATA_DIR, 'mreserve_data', '{}{:03d}of{:03d}.tfrecord'.format(args.split, args.fold, args.num_folds))

split_fn = {
    'train': os.path.join(DATA_DIR, 'json/train_data.json'),
    'val': os.path.join(DATA_DIR, 'json/val_data.json'),
    'test': os.path.join(DATA_DIR, 'json/test_data.json'),
}[args.split]

data = []

data2 = json.load(open(os.path.join(DATA_DIR, 'json/objects.json'), 'r'))
json_data = json.load(open(split_fn, 'r'))

for pair in json_data:
    v1, v2 = pair.split("_")
    for q in json_data[pair]:
        data.append({
            'obj1': v1,
            'obj2': v2,
            'qid': q,
            'question': json_data[pair][q]['text'],
            'label': json_data[pair][q]['label'],
        })

def parse_item(item):

    frames_path1 = os.path.join(DATA_DIR, 'frames486', item['obj1'])
    num_frames1 = max([int(x.split('.')[0]) for x in os.listdir(frames_path1)])
    frames_path2 = os.path.join(DATA_DIR, 'frames486', item['obj2'])
    num_frames2 = max([int(x.split('.')[0]) for x in os.listdir(frames_path2)])

    midframe1 = Image.open(os.path.join(DATA_DIR, "frame_with_box", f"{item['obj1']}.png"))
    midframe1 = resize_image(midframe1, shorter_size_trg=450, longer_size_max=800)
    midframe2 = Image.open(os.path.join(DATA_DIR, "frame_with_box", f"{item['obj2']}.png"))
    midframe2 = resize_image(midframe2, shorter_size_trg=450, longer_size_max=800)

    audio_fn1 = os.path.join(os.path.join(DATA_DIR, "audio21992", f"{item['obj1']}.wav"))
    sr1, waveform1 = wavfile.read(audio_fn1, mmap=False)
    waveform1 = waveform1.astype(np.float32)
    waveform1 /= max(np.abs(waveform1).max(), 1.0)

    audio_fn2 = os.path.join(os.path.join(DATA_DIR, "audio21992", f"{item['obj2']}.wav"))
    sr2, waveform2 = wavfile.read(audio_fn2, mmap=False)
    waveform2 = waveform2.astype(np.float32)
    waveform2 /= max(np.abs(waveform2).max(), 1.0)

    frames1 = []
    frames2 = []

    fps1 = round(data2[item['obj1']]['fps'])
    fps2 = round(data2[item['obj2']]['fps'])

    max1 = min(int(num_frames1 - (3.8*fps1 + 1)), int((len(waveform1) - 21992*3.8)/21992*fps1))
    start1 = random.randint(round(min(max(max1//2, 0.8*fps1), max1)), max1)
    
    max2 = min(int(num_frames2 - (3.8*fps2 + 1)), int((len(waveform2) - 21992*3.8)/21992*fps2))
    start2 = random.randint(round(min(max(max2//2, 0.8*fps2), max2)), max2)

    specs1 = []
    specs2 = []

    for j in range(4):
        i = j*fps1 + start1
        frame = Image.open(os.path.join(DATA_DIR, "frames486", item['obj1'], f"{i:06d}.png"))
        frame = resize_image(frame, shorter_size_trg=450, longer_size_max=800)
        frame = pil_image_to_jpgstring(frame)
        frames1.append(frame)

        start_time = i/fps1 - 0.8
        start_idx = int(sr1 * start_time) - 1
        end_idx = int(sr1 * (start_time + 1.6))

        if start_idx < 0:
            wav_ts = np.concatenate([np.zeros(1-start_idx, dtype=np.float32), waveform1[:end_idx]], 0)
        else:
            wav_ts = waveform1[start_idx:end_idx]

        spec = make_spectrogram2(wav_ts, playback_speed=1, sr=22050, pad_size=0)
        assert spec.shape == (60, 65), f"{spec.shape}, {item['obj1']}, {(start_idx, end_idx)}, {len(waveform1)}"
        specs1.append(spec)
        
    for j in range(4):
        i = j*fps2 + start2
        frame = Image.open(os.path.join(DATA_DIR, "frames486", item['obj2'], f"{i:06d}.png"))
        frame = resize_image(frame, shorter_size_trg=450, longer_size_max=800)
        frame = pil_image_to_jpgstring(frame)
        frames2.append(frame)

        start_time = i/fps2 - 0.8
        start_idx = int(sr2 * start_time) - 1
        end_idx = int(sr2 * (start_time + 1.6))

        if start_idx < 0:
            wav_ts = np.concatenate([np.zeros(1-start_idx, dtype=np.float32), waveform2[:end_idx]], 0)
        else:
            wav_ts = waveform2[start_idx:end_idx]
        
        spec = make_spectrogram2(wav_ts, playback_speed=1, sr=22050, pad_size=0)
        assert spec.shape == (60, 65), f"{spec.shape}, {item['obj2']}, {(start_idx, end_idx)}, {len(waveform2)}"
        specs2.append(spec)
    

    magic_number1 = 255.0 / max(np.percentile(np.array(specs1).reshape(-1, 65), 99), 1.0)
    magic_number2 = 255.0 / max(np.percentile(np.array(specs2).reshape(-1, 65), 99), 1.0)
    
    return num_frames1, num_frames2, frames1, frames2, midframe1, midframe2, specs1, specs2, magic_number1, magic_number2

num_written = 0
max_len = 0
with tf.io.TFRecordWriter(out_fn) as tfrecord_writer:
    for i, item in enumerate(data):
        if i % args.num_folds != args.fold:
            continue

        try:
            num_frames1, num_frames2, frames1, frames2, midframe1, midframe2, specs1, specs2, magic_number1, magic_number2 = parse_item(item)
        except:
            print(f"Parsing item with v1: {item['obj1']}, v2: {item['obj2']} failed, possibly because data was not found")
            continue

        query_enc = encoder.encode(item['question']).ids

        feature_dict = {
            'question': int64_list_feature(query_enc),
            'label': int64_feature(item['label']),
            'obj1': bytes_feature(item['obj1'].encode('utf-8')),
            'obj2': bytes_feature(item['obj2'].encode('utf-8')),
            'num_frames1': int64_feature(num_frames1),
            'num_frames2': int64_feature(num_frames2),
            'midframe1': bytes_feature(pil_image_to_jpgstring(midframe1)),
            'midframe2': bytes_feature(pil_image_to_jpgstring(midframe2)),
            'id': bytes_feature((item['obj1'] + "__" + item['obj2'] + "__" + item['qid']).encode('utf-8')),
            'magic_number1': float_list_feature([magic_number1]),
            'magic_number2': float_list_feature([magic_number2]),
        }

        for i, frame in enumerate(frames1):
            feature_dict[f'frames1_{i:03d}'] = bytes_feature(frame)
            compressed = np.minimum(specs1[i].reshape(-1, 65) * magic_number1, 255.0).astype(np.uint8)
            assert compressed.shape == (60, 65)
            feature_dict[f'spec1_{i:03d}'] = bytes_feature(pil_image_to_jpgstring(Image.fromarray(compressed)))
        for i, frame in enumerate(frames2):
            feature_dict[f'frames2_{(i):03d}'] = bytes_feature(frame)
            compressed = np.minimum(specs2[i].reshape(-1, 65) * magic_number2, 255.0).astype(np.uint8)
            assert compressed.shape == (60, 65)
            feature_dict[f'spec2_{i:03d}'] = bytes_feature(pil_image_to_jpgstring(Image.fromarray(compressed)))

        if num_written < 20:
            print("Writing example {}".format(num_written), flush=True)
            print("Question: {}".format(item['question']), flush=True)
            print("Label: {}".format(item['label']), flush=True)
            print("Object 1: {}".format(item['obj1']), flush=True)
            print("Object 2: {}".format(item['obj2']), flush=True)
            print("id: {}".format(item['obj1'] + "__" + item['obj2'] + "__" + item['qid']), flush=True)
            print("query", query_enc, flush=True)

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        tfrecord_writer.write(example.SerializeToString())
    
        num_written += 1
        if num_written % 100 == 0:
            print("Have written {} / {}".format(num_written, len(data)), flush=True)

print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)

