import os
import torch
import numpy as np
import random

class Lunar:
    features = []
    videos = []

    for f in os.listdir('lunar'):
        if not f.startswith('features') or not f.endswith('pt'):
            continue
        f_ = f[9:].split('.')[0]
        videos.append(f'https://mixturevideos.s3-us-west-2.amazonaws.com/lunar/video-{f_}.mp4')
        features.append(torch.load(f'lunar/features-{f_}.pt'))

    print(f'Loaded {len(videos)} lunar trajectories')

    features = torch.tensor(np.stack(features)).float()
    f_tup = [tuple(x) for x in features]
    assert len(f_tup) == len(set(f_tup))

    messages = ['Land softly on the landing pad', 'Stay in the air as long as possible']


class Fetch:
    features = []
    videos = []

    for f in os.listdir('fetch'):
        if not f.startswith('features') or not f.endswith('pt'):
            continue
        f_ = f[9:].split('.')[0]
        videos.append(f'https://mixturevideos.s3-us-west-2.amazonaws.com/fetch/{f_}.webm')
        features.append(torch.load(f'fetch/features-{f_}.pt'))

    print(f'Loaded {len(videos)} fetch trajectories')

    features = torch.tensor(np.stack(features)).float()
    f_tup = [tuple(x) for x in features]
    assert len(f_tup) == len(set(f_tup))

    messages = ['Rank trajectories'] 
