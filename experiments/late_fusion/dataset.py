import os
from torch.utils.data import Dataset
import json
import cv2
from PIL import Image, ImageFile
import clip
import numpy as np
from numpy.random import randint
import torchaudio
import torch
import random

torchaudio.set_audio_backend("soundfile")
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PACSFusionDataset(Dataset):
    def __init__(self, root_path, split, audio_conf, use_audio=True, use_img=True, use_vid=True, img_transform=None, q_transform=None, v_transform=None, num_segments=8, new_length=5, image_tmpl='{:06d}.png', random_shift=True, test_mode=False):
        self.images = {}
        self.audios = {}
        self.root_path = root_path
        self.split = split
        self.img_transform = img_transform
        self.q_transform = q_transform
        self.v_transform = v_transform
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.audio_conf = audio_conf
        if self.audio_conf:
            self.melbins = self.audio_conf.get('num_mel_bins')
            self.freqm = self.audio_conf.get('freqm')
            self.timem = self.audio_conf.get('timem')
            self.norm_mean = self.audio_conf.get('mean')
            self.norm_std = self.audio_conf.get('std')
            self.noise = self.audio_conf.get('noise')
        self.data = self.load_data()
        self.length = len(self.data)
        self.use_audio = use_audio
        self.use_img = use_img
        self.use_vid = use_vid

    def _get_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(directory, self.image_tmpl.format(1))).convert('RGB')]

    def _sample_indices(self, video_list):

        if((len(video_list) - self.new_length + 1) < self.num_segments):
            average_duration = (len(video_list) - 5 + 1) // (self.num_segments)
        else:
            average_duration = (len(video_list) - self.new_length + 1) // (self.num_segments)
        offsets = []
        if average_duration > 0:
            offsets += list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,size=self.num_segments))
        elif len(video_list) > self.num_segments:
            if((len(video_list) - self.new_length + 1) >= self.num_segments):
                offsets += list(np.sort(randint(len(video_list) - self.new_length + 1, size=self.num_segments)))
            else:
                offsets += list(np.sort(randint(len(video_list) - 5 + 1, size=self.num_segments)))
        else:
            offsets += list(np.zeros((self.num_segments,)))
        offsets = np.array(offsets)
        return offsets + 1

    def _get_val_indices(self, video_list):
        if len(video_list) > self.num_segments + self.new_length - 1:
            tick = (len(video_list) - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, video_list):
        if len(video_list) > self.num_segments + self.new_length - 1:
            tick = (len(video_list) - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def __len__(self):
        return self.length

    def get_vid(self, path, video_list, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(0,self.new_length,1):
                seg_imgs = self._get_image(path, p)
                images.extend(seg_imgs)
                if((len(video_list)-self.new_length*1+1)>=8):
                    if p < (len(video_list)):
                        p += 1
                else:
                    if p < (len(video_list)):
                        p += 1

        process_data, _ = self.v_transform((images, "0"))

        return process_data

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank
    
    def _load_audio(self, sample_obj):
        fbank = self._wav2fbank(os.path.join(self.root_path, "audio16000", sample_obj + ".wav"))
        freq = random.randint(0, self.freqm)
        freqm = torchaudio.transforms.FrequencyMasking(freq)
        time = random.randint(0, self.timem)
        timem = torchaudio.transforms.TimeMasking(time)
        fbank = torch.transpose(fbank, 0, 1)

        if freq != 0:
            fbank = freqm(fbank)
        if time != 0:
            fbank = timem(fbank)

        fbank = torch.transpose(fbank, 0, 1)

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10

        return fbank

    def _load_video(self, sample_obj):
        path = os.path.join(self.root_path,"frames252", sample_obj)
        video_list = os.listdir(path)

        if not self.test_mode:
            segment_indices = self._sample_indices(video_list) if self.random_shift else self._get_val_indices(video_list) 
        else:
            segment_indices = self._get_test_indices(video_list)

        vid_data = self.get_vid(path, video_list, segment_indices)

        return vid_data

    def _load_image(self, sample_obj):
        img = Image.open(os.path.join(self.root_path,"center_with_box", sample_obj + ".png"))
        img = self.img_transform(img)
        return img

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.use_vid:
            vid_data1 = self._load_video(sample['obj1'])
            vid_data2 = self._load_video(sample['obj2'])
        else:
            vid_data1 = torch.tensor([])
            vid_data2 = torch.tensor([])

        if self.use_audio:
            audio_data1 = self._load_audio(sample['obj1'])
            audio_data2 = self._load_audio(sample['obj2'])
        else:
            audio_data1 = torch.tensor([])
            audio_data2 = torch.tensor([])

        img_data1 = self._load_image(sample['obj1'])
        img_data2 = self._load_image(sample['obj2'])

        target = {"obj1":sample["obj1"], "obj2":sample["obj2"], "question":sample["question"], "label":sample["label"], "img1":img_data1, "img2":img_data2, "vid1": vid_data1, "vid2": vid_data2, "audio1": audio_data1, "audio2": audio_data2, "id":sample["id"]}

        if self.q_transform:
            target = self.q_transform(target)

        return target

    def load_data(self):
        data_json = json.load(open(os.path.join(self.root_path, 'json', self.split + '.json')))
        data = []

        for pair in data_json:
            v1, v2 = pair.split("_")

            if not (os.path.exists(os.path.join(self.root_path,"center_with_box", v1 + ".png"))):
                print(f"did not find image: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.root_path,"center_with_box", v2 + ".png"))):
                print(f"did not find image: {v2}")
                continue
            if not (os.path.exists(os.path.join(self.root_path,"audio16000", v1 + ".wav"))):
                print(f"did not find audio: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.root_path,"audio16000", v2 + ".wav"))):
                print(f"did not find audio: {v2}")
                continue
            if not (os.path.exists(os.path.join(self.root_path,"frames252", v1 ))):
                print(f"did not find video: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.root_path,"frames252", v2 ))):
                print(f"did not find video: {v2}")
                continue
 
            for q in data_json[pair]:
                sample = {"obj1":v1, "obj2":v2, "question":data_json[pair][q]["text"], "label":data_json[pair][q]["label"], "id": q}
                data.append(sample)

        return data