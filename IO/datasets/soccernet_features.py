import warnings
from abc import abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from random import sample

from .soccernet import SoccerNet
from .soccernet import SoccerNetLabelsLoader
from .soccernet import Stream


class VectorFeatures:
    def __init__(self, name, fname_template, dim):
        self.name = name
        self.fname_template = fname_template
        self.dim = dim

    def filename(self, half):
        return self.fname_template.format(half)


class FeatureStream(Stream):
    _Features = {'resNet152-PCA': VectorFeatures('resNet152-PCA', '{}_ResNET_TF2_PCA512.npy', 512),
                 'vggish': VectorFeatures('vggish', '{}_VGGish.npy', 512),
                 'resNet152': VectorFeatures('resNet152', '{}_ResNET_TF2.npy', 2048),
                 'baidu': VectorFeatures('baidu', '{}_baidu_soccer_embeddings.npy', 8576),
                 'DynamicEdgeConvGCN': VectorFeatures('DynamicEdgeConvGCN',
                                                      'NetVLAD++_graph:DynamicEdgeConvGCN bboxes:pointrend '
                                                      'calibrations: ccbv feature_size:12 layer:fc1 vector_size:512 '
                                                      'half:{}.npy', 512)
                 }

    def __init__(self, name, features_name):
        super().__init__(name)
        self.features = FeatureStream._Features[features_name]

    def __repr__(self):
        return f'(FeatureStream: {self.name}, Features: {self.features.name})'


class SoccerNetFeaturesBase(SoccerNet):
    def __init__(self, data_dir: Path, splits: List[str], feature_streams: List[FeatureStream], **kwargs):
        super().__init__(data_dir, splits, **kwargs)
        self.feature_streams = feature_streams

    def _load_features(self, match_path, half, stride, off=0):
        num_batches, features = set(), dict()

        # Loading features by modality
        full_match_path = self.path.joinpath(match_path)
        for s in self.feature_streams:
            features_fname = s.features.filename(half + 1)
            features_path = full_match_path.joinpath(features_fname)

            stream_features = np.load(features_path).astype(np.float32)
            num_dim = stream_features.shape[-1]
            stream_features = stream_features.reshape(-1, num_dim)
            if s.name == 'graph':
                stream_features = stream_features[::stride]
                assert not np.isnan(stream_features).any(), f'NAN in match {match_path}'
                stream_features = torch.from_numpy(stream_features)
            else:
                stream_features = SoccerNet.to_clip(torch.from_numpy(stream_features),
                                                    stride, self.frames_per_window, off)
            features[s.name] = stream_features
            num_batches.add(stream_features.shape[0])

        if len(num_batches) != 1:
            warnings.warn(f'Number of batches differ for match {match_path} and half {half}')
            num_batches = min(num_batches)
            for s in self.feature_streams:
                features[s.name] = features[s.name][:num_batches, :]
        else:
            num_batches = num_batches.pop()
        return features, num_batches

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SoccerNetFeatures(SoccerNetFeaturesBase, SoccerNetLabelsLoader):
    def __init__(self, data_dir: Path, splits: List[str], feature_streams: List[FeatureStream], window_size_sec=60,
                 frame_rate=2, tiny=None, num_random_samples=None):
        super().__init__(data_dir, splits, feature_streams, window_size_sec=window_size_sec, frame_rate=frame_rate)

        self.features = {s.name: [] for s in self.feature_streams}

        half_matches = self.half_matches[:tiny]
        if num_random_samples:
            half_matches = sample(half_matches, k=min(num_random_samples, len(half_matches)))

        for match_path, half in tqdm(half_matches):
            features, num_batches = self._load_features(match_path, half, self.frames_per_window)
            for s in self.feature_streams:
                self.features[s.name].append(features[s.name])

            if 'challenge' not in self.splits:
                self._load_labels(match_path, half, num_batches)

        for s in self.feature_streams:
            self.features[s.name] = np.concatenate(self.features[s.name])

        if 'challenge' not in self.splits:
            self.labels = np.concatenate(self.labels)

        self.features_dim = {s: self.features[s.name].shape[-1] for s in self.feature_streams}

    def __getitem__(self, index):
        features = [self.features[s.name][index, ...] for s in self.feature_streams]
        labels = self.labels[index, :] if 'challenge' not in self.splits else None
        return features, labels

    def __len__(self):
        return len(self.features[self.feature_streams[0].name])


class SoccerNetFeaturesTesting(SoccerNetFeaturesBase):
    def __init__(self, data_dir: Path, splits: List[str], feature_streams: List[FeatureStream], window_size_sec=60,
                 frame_rate=2, tiny=None):
        super().__init__(data_dir, splits, feature_streams, window_size_sec=window_size_sec, frame_rate=frame_rate)

        if tiny:
            assert tiny % 2 == 0, "tiny variable must be an even integer"

        self.half_matches = self.half_matches[:tiny]
        self.matches = [hm[0] for hm in self.half_matches[::2]]

    def __getitem__(self, index):
        match_path = self.matches[index]
        features = []
        for half in range(2):
            f, _ = self._load_features(match_path, half, 1, self.frames_per_window // 2)
            features.append([f[s.name] for s in self.feature_streams])
        return str(match_path), features[0], features[1]

    def __len__(self):
        return len(self.matches)
