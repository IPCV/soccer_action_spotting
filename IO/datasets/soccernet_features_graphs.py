import warnings
from abc import abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from .soccernet import SoccerNet
from .soccernet import Stream, SoccerNetLabelsLoader
from .soccernet_features import FeatureStream
from .soccernet_graphs import GraphStream
from .soccernet_graphs import SoccerNetGraphBase


class SoccerNetMultimodalBase(SoccerNetGraphBase):
    def __init__(self, data_dir: Path, splits: List[str], streams: List[Stream], **kwargs):
        self.streams = streams
        graph_stream = None
        for s in streams:
            if isinstance(s, GraphStream):
                graph_stream = s
                break
        super().__init__(data_dir, splits, graph_stream=graph_stream, **kwargs)

    def _load_features(self, match_path, half, half_match_graph, stride, off=0):
        num_batches, features = set(), dict()

        # Loading features by modality
        full_match_path = self.path.joinpath(match_path)
        for s in self.streams:
            if isinstance(s, FeatureStream):
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

                num_batches.add(stream_features.shape[0])
            else:
                node_features, edges = half_match_graph['features'], half_match_graph['edges']
                stream_features = self._half_match_to_graph(node_features, edges, self.frames_per_window)
                num_batches.add(len(stream_features))
            features[s.name] = stream_features

        if len(num_batches) != 1:
            warnings.warn(f'Number of batches differ for match {match_path} and half {half}')
            num_batches = min(num_batches)
            for s in self.streams:
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


class SoccerNetMultimodal(SoccerNetMultimodalBase, SoccerNetLabelsLoader):
    def __init__(self, data_dir: Path, splits: List[str], streams: List[Stream], window_size_sec=60, frame_rate=2,
                 distance_players=25, invert_second_half_positions=True, filter_positions=True, min_detection_score=0,
                 use_velocity=True, use_player_prediction=True, calibration_flags=None, tiny=None, cache=True,
                 overwrite_cache=False):

        super().__init__(data_dir, splits, streams, window_size_sec=window_size_sec, frame_rate=frame_rate,
                         distance_players=distance_players, invert_second_half_positions=invert_second_half_positions,
                         filter_positions=filter_positions, min_detection_score=min_detection_score,
                         use_velocity=use_velocity, use_player_prediction=use_player_prediction,
                         calibration_flags=calibration_flags)

        self.features = {s.name: [] for s in self.streams}
        match_graphs = self.load_graphs(cache, overwrite_cache, tiny)

        for match_path, half in tqdm(self.half_matches[:tiny]):

            half_match_graph = match_graphs[match_path][half]
            features, num_batches = self._load_features(match_path, half, half_match_graph, self.frames_per_window)

            for s in self.streams:
                if isinstance(s, FeatureStream):
                    self.features[s.name].append(features[s.name])
                else:
                    self.features[s.name].extend(features[s.name])

            if 'challenge' not in self.splits:
                self._load_labels(match_path, half, num_batches)

        for s in self.streams:
            if isinstance(s, FeatureStream):
                self.features[s.name] = np.concatenate(self.features[s.name])

        if 'challenge' not in self.splits:
            self.labels = np.concatenate(self.labels)

        self.features_dim = {s: self.features[s.name].shape[-1] for s in self.feature_streams
                             if isinstance(s, FeatureStream)}

    def __getitem__(self, index):
        features = [self.features[s.name][index, ...] for s in self.streams]
        labels = torch.from_numpy(self.labels[index, :]) if 'challenge' not in self.splits else None
        return features, labels

    def __len__(self):
        return len(self.features[self.streams[0].name])

    def collate(self):
        def _collate_challenge(examples: List):
            return Batch.from_data_list([x for b in examples for x in b[0]])

        def _collate(examples: List):
            return Batch.from_data_list([x for b in examples for x in b[0]]), \
                   torch.stack([x[1] for x in examples], dim=0)

        return _collate_challenge if 'challenge' in self.splits else _collate


class SoccerNetMultimodalTesting(SoccerNetMultimodalBase):

    def __init__(self, data_dir: Path, splits: List[str], streams: List[Stream], window_size_sec=60, frame_rate=2,
                 distance_players=25, invert_second_half_positions=True, filter_positions=True, min_detection_score=0,
                 use_velocity=True, use_player_prediction=True, calibration_flags=None, tiny=None, cache=True,
                 overwrite_cache=False):

        super().__init__(data_dir, splits, streams, window_size_sec=window_size_sec, frame_rate=frame_rate,
                         distance_players=distance_players, invert_second_half_positions=invert_second_half_positions,
                         filter_positions=filter_positions, min_detection_score=min_detection_score,
                         use_velocity=use_velocity, use_player_prediction=use_player_prediction,
                         calibration_flags=calibration_flags)

        if tiny:
            assert tiny % 2 == 0, "tiny variable must be an even integer"

        self.match_graphs = self.load_graphs(cache, overwrite_cache, tiny)

        self.half_matches = self.half_matches[:tiny]
        self.matches = [hm[0] for hm in self.half_matches[::2]]

    def __getitem__(self, index):
        match_path = self.matches[index]
        features = []
        for half in range(2):
            half_match_graph = self.match_graphs[match_path][half]
            f, _ = self._load_features(match_path, half, half_match_graph, 1, self.frames_per_window // 2)
            features.append([f[s.name] for s in self.streams])
        return str(match_path), features[0], features[1]

    def __len__(self):
        return len(self.matches)

    @staticmethod
    def collate(examples: List):
        output = [[b[0] for b in examples]]

        for half in range(2):
            print(half)

        return [b[0] for b in examples], \
               Batch.from_data_list([c for b in examples for x in b[1] for c in x]), \
               Batch.from_data_list([c for b in examples for x in b[2] for c in x])