import inspect
import json
import os
import sys
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from addict import Dict

# TODO: Remove this hackish way of loading parent's dir modules
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from IO.preprocess import VGGishPreprocess
from models import VGGishFeatures
from tqdm import tqdm


def extract_features(wav_file, preprocess, vgg):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    delta_time, num_secs = sr // 2, 60
    start, cur_seg, total_seg = 0, 0, int(len(wav_data) / delta_time)
    features = np.zeros((total_seg, 512), dtype=np.float32)

    while cur_seg < total_seg:
        data = np.zeros((num_secs, 1, 46, 64))
        for i in range(num_secs):
            if start + delta_time > len(wav_data):
                break
            cur_wav = wav_data[start:start + delta_time]
            data[i, :, :, :] = preprocess(cur_wav, sr, return_tensor=False)
            start += delta_time
            cur_seg += 1
        data = torch.tensor(data, requires_grad=True).float()
        current_features = vgg(data)
        current_features = current_features.cpu().detach().numpy()
        features[cur_seg - current_features.shape[0]:cur_seg, :] = current_features
    return features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', required=False,
                        help='JSON audio extraction conf',
                        default="config/audio_extraction.json", type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--GPU', required=False,
                        help='ID of the GPU to use',
                        default=-1, type=int)
    args = parser.parse_args()

    with open(args.conf) as json_file:
        sampling_params = Dict(json.load(json_file))
    preprocess = VGGishPreprocess(sampling_params)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    model_urls = {
        'vggish': 'https://github.com/harritaylor/torchvggish/'
                  'releases/download/v0.1/vggish-10086976.pth'
    }
    vgg = VGGishFeatures(model_urls, pretrained=True, preprocess=None, progress=True)

    wav_regex = str(args.dataset_path.joinpath('**/[12].wav'))
    wav_files = [Path(f) for f in glob(wav_regex, recursive=True)]

    for wav_file in tqdm(wav_files):
        parent_dir = wav_file.parent
        np_fpath = parent_dir.joinpath(wav_file.stem + '_VGGish.npy')

        features = extract_features(wav_file, preprocess, vgg)
        np.save(np_fpath, features)
