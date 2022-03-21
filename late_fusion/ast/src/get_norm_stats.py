
import torch
import numpy as np

import dataloader

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 48, 'timem': 96, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset'}

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset('YOUR_DATA_HERE', label_csv='LABEL_CSV_HERE',
                                audio_conf=audio_conf), batch_size=50, shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    # print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))