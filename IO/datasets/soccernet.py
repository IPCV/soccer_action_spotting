import csv
from abc import ABC, abstractmethod
from math import ceil
from operator import mul
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class SoccerNet(ABC, data.Dataset):

    def __init__(self, data_dir: Path, splits: List[str], window_size_sec=60, frame_rate=2, version='v2', **kwargs):

        self.path = data_dir
        self.version = version

        videos_csv_path = self.path.joinpath(f'videos_{self.version}.csv')
        self.videos = SoccerNet.read_videos(videos_csv_path)

        classes_csv_path = self.path.joinpath(f'classes_{self.version}.csv')
        self.classes = SoccerNet.read_classes(classes_csv_path)
        self.num_classes = len(self.classes)

        if isinstance(splits, str):
            splits = [splits]
        self.splits = splits

        csv_paths = [self.path.joinpath(f'action_spotting_{s}_{self.version}.csv') for s in self.splits]
        if 'challenge' not in self.splits:
            self.annotations = pd.concat([SoccerNet.read_annotations(a) for a in csv_paths])
            self.half_matches = [(mp, h) for mp, h in self.annotations.index.unique()]
        else:
            self.annotations = None
            matches = pd.concat([SoccerNet.read_challenge(c) for c in csv_paths])
            self.half_matches = [(mp, h) for mp, h in zip(matches['match_path'], range(2))]

        self.window_size_sec = window_size_sec
        self.frame_rate = frame_rate
        self.frames_per_window = self.window_size_sec * self.frame_rate

    def _load_labels(self, match_path, half, num_batches):
        pass

    @staticmethod
    def read_classes(classes_csv_path):
        with open(classes_csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            return {i: r[0] for i, r in enumerate(csv_reader)}

    @staticmethod
    def read_videos(videos_csv_path):
        return pd.read_csv(videos_csv_path,
                           usecols=['match_path',
                                    'match_date',
                                    'visiting_team',
                                    'home_team',
                                    'score',
                                    'first_half_duration_sec',
                                    'second_half_duration_sec'],
                           dtype={'match_date': str,
                                  'visiting_team': str,
                                  'home_team': str,
                                  'score': str},
                           converters={'match_path': Path,
                                       'first_half_duration_sec': lambda d: int(float(d)),
                                       'second_half_duration_sec': lambda d: int(float(d))
                                       })

    @staticmethod
    def read_annotations(annotations_csv_path):
        to_secs = lambda t: sum(map(mul, [60, 1], map(int, t.split(':'))))
        return pd.read_csv(annotations_csv_path,
                           usecols=['match_path',
                                    'half',
                                    'game_time',
                                    'label',
                                    'position',
                                    'team',
                                    'visibility'],
                           dtype={'label': int,
                                  'position': int,
                                  'team': int
                                  },
                           converters={'match_path': Path,
                                       'half': lambda h: int(h) - 1,
                                       'game_time': to_secs,
                                       'visibility': lambda v: 1 if int(v) else -1},
                           index_col=['match_path',
                                      'half'])

    @staticmethod
    def read_challenge(challenge_csv_path):
        return pd.read_csv(challenge_csv_path,
                           usecols=['match_path'],
                           converters={'match_path': Path})

    @staticmethod
    def to_clip(features, stride, clip_length, off=0, padding="replicate_last"):
        num_features = len(features) if isinstance(features, list) else features.shape[0]

        if padding == "zeropad":
            # FIXME: Consider also lists
            pad = torch.nn.ZeroPad2d((0, 0, clip_length - num_features % stride, 0))
            features = pad(features)

        idx = torch.tile(torch.arange(start=0, end=num_features - 1, step=stride), (clip_length, 1)).t()
        idx += torch.tile(torch.arange(-off, clip_length - off), (ceil((num_features - 1) / stride), 1))

        if padding == "replicate_last":
            idx = idx.clamp(0, num_features - 1)

        if isinstance(features, list):
            return [[features[i] for i in id_i] for id_i in idx.tolist()]
        else:
            return features[idx, :]

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SoccerNetLabelsLoader(SoccerNet):
    background_class = 0

    def __init__(self, data_dir: Path, splits: List[str], **kwargs):
        super().__init__(data_dir, splits, **kwargs)
        self.labels = []

    def _load_labels(self, match_path, half, num_batches):
        labels = np.zeros((num_batches, self.num_classes))
        labels[:, SoccerNetLabelsLoader.background_class] = 1
        for r in self.annotations.loc[(match_path, half)].itertuples():
            index = r.game_time // self.window_size_sec
            if index < num_batches:
                labels[index, SoccerNetLabelsLoader.background_class] = 0
                labels[index, r.label] = 1
        self.labels.append(labels)


class SoccerNetStats(SoccerNet):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir, ['train', 'valid', 'test'], 0, 0)

        csv_paths = [self.path.joinpath(f'action_spotting_{s}_{self.version}.csv') for s in self.splits]

        self.annotations_by_split = {s: SoccerNet.read_annotations(a) for s, a in zip(self.splits, csv_paths)}
        self.matches = list(set([hm[0] for hm in self.half_matches]))
        self.histograms = SoccerNetStats.calculate_histograms(self)

    @staticmethod
    def calculate_histograms(dataset):
        histograms = np.zeros((dataset.num_classes - 1, len(dataset.splits) + 1), dtype=np.int32)
        for i, s in enumerate(dataset.splits):
            labels = dataset.annotations_by_split[s].label.to_numpy()
            histograms[:, i] = np.histogram(labels, bins=dataset.num_classes - 1)[0]
        histograms[:, -1] += np.sum(histograms[:, :-1], axis=1)
        return histograms

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


class Stream(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __repr__(self):
        return f'(Stream: {self.name})'
