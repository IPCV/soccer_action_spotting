import csv
import json
import logging
from abc import abstractmethod
from functools import reduce
from itertools import zip_longest
from operator import truediv
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.special import softmax
from torch_geometric.data import Batch
from torch_geometric.data import Data
from tqdm import tqdm

from field_calibration import calculate_player_position
from field_calibration import load_homography
from field_calibration import unproject_image_point
from .soccernet import SoccerNet
from .soccernet import SoccerNetLabelsLoader
from .soccernet import Stream


class Predictions:
    def __init__(self, name, fname_template):
        self.name = name
        self.fname_template = fname_template

    def filename(self, half):
        return self.fname_template.format(half)

    def __repr__(self):
        return f'(Predictions: {self.name}, fname_template: {self.fname_template})'


class GraphStream(Stream):
    _Predictions = {'maskrcnn': Predictions('maskrcnn', '{}_player_boundingbox_maskrcnn.json'),
                    'pointrend': Predictions('pointrend', 'player_velocity_team_results.npy'),
                    'ccbv': Predictions('ccbv', '{}_field_calib_ccbv.json')}

    def __init__(self, name, bboxes_name, calibrations_name, use_player_prediction=True, use_velocity=True):
        self.bboxes = GraphStream._Predictions[bboxes_name]
        self.calibrations = GraphStream._Predictions[calibrations_name]
        self.feature_vector_size = 12 if self.bboxes.name == 'pointrend' else 8
        if not use_player_prediction:
            self.feature_vector_size -= 5
        if not use_velocity:
            self.feature_vector_size -= 2
        super().__init__(f'{name} ' +
                         f'bboxes:{self.bboxes.name} ' +
                         f'calibrations: {self.calibrations.name} ' +
                         f'feature_size:{self.feature_vector_size}')

    def __repr__(self):
        return f'(GraphStream: {self.name})'


class SoccerNetGraphBase(SoccerNet):
    def __init__(self, data_dir: Path, splits: List[str], graph_stream: GraphStream, **kwargs):
        super().__init__(data_dir, splits, **kwargs)

        props = {
            'distance_players': 25,
            'invert_second_half_positions': True,
            'filter_positions': True,
            'min_detection_score': 0,
            'use_velocity': True,
            'use_player_prediction': True,
            'calibration_flags': None,
            'calibration_threshold': 0.75
        }
        props.update(kwargs)

        self.graph_stream = graph_stream

        self.calibration_flags = props['calibration_flags']
        self.calibration_threshold = props['calibration_threshold']

        self.max_dist_player = props['distance_players']
        self.invert_second_half_positions = props['invert_second_half_positions']
        self.filter_positions = props['filter_positions']
        self.min_detection_score = props['min_detection_score']
        self.use_velocity = props['use_velocity']
        self.use_player_prediction = props['use_player_prediction']

        self.feature_vector_size = self.graph_stream.feature_vector_size
        if self.calibration_flags.use_confidence:
            self.feature_vector_size += 1

        sars_filepath = self.path.joinpath('sampling_aspect_ratio.csv')
        with sars_filepath.open(mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            self.sampling_aspect_ratios = {
                self.path.joinpath(r["match_path"]): reduce(truediv, map(float, r["SAR"].split(':'))) for r in
                csv_reader}

    def _load_bboxes(self, match_path, half):
        if self.graph_stream.bboxes.name == 'pointrend':
            numpy_fpath = self.__build_prediction_path(match_path, half, predictions=self.graph_stream.bboxes)
            data = np.load(numpy_fpath, allow_pickle=True).item()
            return data[half]
        else:
            data = self.__load_jsons(match_path, half, predictions=self.graph_stream.bboxes)
            return data['predictions']

    def _load_calibration(self, match_path, half):
        data = self.__load_jsons(match_path, half, predictions=self.graph_stream.calibrations)
        homographies = []
        for prediction in data['predictions']:
            homographies.append({'matrix': prediction[0]['homography'],
                                 'confidence': prediction[0]['confidence']})
        return homographies, data['size']

    def _cache_path(self, tiny=None):
        fname = f'cache_players_graphs ' + \
                f'bboxes:{self.graph_stream.bboxes.name} ' + \
                f'splits:{"_".join(self.splits)} ' + \
                f'frames_per_window:{self.frames_per_window} ' + \
                f'max_dist_player:{self.max_dist_player} ' + \
                f'use_calibration:{self.calibration_flags.use_calibration} ' + \
                f'use_confidence:{self.calibration_flags.use_confidence} ' + \
                f'use_cone:{self.calibration_flags.use_calibration_cone} ' + \
                f'invert_second_half_positions:{self.invert_second_half_positions} ' + \
                f'filter_positions:{self.filter_positions} ' + \
                f'min_detection_score:{self.min_detection_score}  '
        fname += f' tiny:{tiny}.npz' if tiny else '.npz'
        return self.path.joinpath(fname)

    @staticmethod
    def to_node_features(bbox, color, velocity, team, position, frame_center, calibration_confidence):
        node_features = []
        if color is not None:
            node_features.extend([c / 127 - 1 for c in color[:3]])

        node_features.append(float(bbox[0] - bbox[2]) * float(bbox[1] - bbox[3]) / 10000)  # Area

        if velocity is not None:
            node_features.extend(velocity)

        if team is not None:
            node_features.extend(softmax(team))

        node_features.extend([p / 50 for p in position[:2]])
        node_features.extend([c / 50 for c in frame_center[:2]])

        if calibration_confidence is not None:
            node_features.append(calibration_confidence)
        return np.array(node_features, dtype=np.float32)

    def load_frame_graph(self, players, homography, scaling_matrix, calibration_size, half):
        homography_matrix = None
        frame_center_projected = [0, 0, 1]
        if self.calibration_flags.use_calibration and homography['confidence'] > self.calibration_threshold:
            homography_matrix = load_homography(homography['matrix']) @ scaling_matrix

            if self.calibration_flags.use_calibration_cone:
                frame_center_projected = unproject_image_point(homography_matrix, np.array(
                    [calibration_size[2] / 2, calibration_size[1] / 2, 1]))

        if self.calibration_flags.use_calibration and homography_matrix is None:
            return [], np.array([], np.uint8)

        if self.filter_positions:
            for bb in players['bboxes']:
                position = calculate_player_position(bb, homography_matrix, self.calibration_flags.use_calibration)
                if not (-100 < position[0] < 100 and -150 < position[0] < 150):
                    return [], np.array([], np.uint8)

        calibration_confidence = None
        if self.calibration_flags.use_calibration and self.calibration_flags.use_confidence:
            calibration_confidence = homography['confidence']

        positions, edges, features = [], [], []
        player_properties = map(lambda a: players.get(a, []), ['bboxes', 'colors', 'scores', 'velocities', 'teams'])
        for i, (bb, color, score, velocity, team) in enumerate(zip_longest(*player_properties, fillvalue=None)):
            if score is not None and score < self.min_detection_score:
                continue

            position = calculate_player_position(bb, homography_matrix, self.calibration_flags.use_calibration)
            if self.invert_second_half_positions and half == 1:
                position[0] *= -1

            if self.calibration_flags.use_calibration:
                for j, other_pos in enumerate(positions):
                    dist = np.linalg.norm(other_pos - position)
                    if dist < self.max_dist_player:
                        edges.append([i, j])
                        edges.append([j, i])

            if not self.use_velocity:
                velocity = None

            if not self.use_player_prediction:
                team = None

            # keep track of all poses
            positions.append(position)
            features.append(SoccerNetGraphBase.to_node_features(bb, color, velocity, team, position,
                                                                frame_center_projected, calibration_confidence))
        return features, np.array(edges, np.uint8)

    def load_match_graph(self, match_path, half):
        players = self._load_bboxes(match_path, half)
        homographies, calibration_size = self._load_calibration(match_path, half)

        scaling_matrix = np.identity(3)
        sar = self.sampling_aspect_ratios[self.path.joinpath(match_path)]
        if sar > 1:
            scaling_matrix[0, 0] = 1 / sar

        num_total_frames = max(len(players), len(homographies))
        num_data_frames = min(len(players), len(homographies))

        half_match_features, half_match_edges = [], []
        for i in range(num_total_frames):
            if i < num_data_frames and players[i]:
                features, edges = self.load_frame_graph(players[i], homographies[i], scaling_matrix, calibration_size,
                                                        half)
            else:
                features, edges = [], []

            # Not players detected in a frame
            if len(features) == 0:
                features.append(np.zeros(self.feature_vector_size, dtype=np.float32))

            half_match_features.append(features)
            half_match_edges.append(edges)
        return half_match_features, half_match_edges

    def load_graphs(self, cache, overwrite_cache, tiny=None):
        cache_path = self._cache_path(tiny) if cache else None
        if cache and overwrite_cache and cache_path.is_file():
            cache_path.unlink()

        match_graphs = None
        if cache and cache_path.is_file():
            logging.info("Loading cache file")
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    match_graphs = data['match_graphs'].item()
            except:
                logging.info("Failed to load cache file")
                match_graphs = None

        if not match_graphs:
            logging.info("Pre-computing clips")
            match_graphs = {}
            for m, h in tqdm(self.half_matches[:tiny]):
                features, edges = self.load_match_graph(m, h)
                if m not in match_graphs:
                    match_graphs[m] = {}
                match_graphs[m][h] = {"features": features, "edges": edges}

            if cache:
                np.savez_compressed(cache_path, match_graphs=match_graphs)

        return match_graphs

    def _half_match_to_graph(self, node_features, edges, stride, off=0):
        representation = []
        num_frames = len(node_features)
        for i in range(num_frames):
            edge_index = torch.tensor(edges[i], dtype=torch.long)
            x = torch.tensor(node_features[i], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            representation.append(data)
        return SoccerNet.to_clip(representation, stride, self.frames_per_window, off)

    def __build_prediction_path(self, match_path, half, predictions):
        full_match_path = self.path.joinpath(match_path)
        prediction_fname = predictions.filename(half + 1)
        return full_match_path.joinpath(prediction_fname)

    def __load_jsons(self, match_path, half, predictions):
        json_path = self.__build_prediction_path(match_path, half, predictions)
        with json_path.open() as json_file:
            return json.load(json_file)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SoccerNetGCN(SoccerNetGraphBase, SoccerNetLabelsLoader):
    def __init__(self, data_dir: Path, splits: List[str], graph_stream: GraphStream, window_size_sec=60, frame_rate=2,
                 distance_players=25, invert_second_half_positions=True, filter_positions=True, min_detection_score=0,
                 use_velocity=True, use_player_prediction=True, calibration_flags=None, tiny=None, cache=True,
                 overwrite_cache=False):

        super().__init__(data_dir, splits, graph_stream, window_size_sec=window_size_sec, frame_rate=frame_rate,
                         distance_players=distance_players, invert_second_half_positions=invert_second_half_positions,
                         filter_positions=filter_positions, min_detection_score=min_detection_score,
                         use_velocity=use_velocity, use_player_prediction=use_player_prediction,
                         calibration_flags=calibration_flags)
        self.players = []

        match_graphs = self.load_graphs(cache, overwrite_cache, tiny)

        for match_path, half in tqdm(self.half_matches[:tiny]):
            node_features = match_graphs[match_path][half]['features']
            edges = match_graphs[match_path][half]['edges']

            players = self._half_match_to_graph(node_features, edges, self.frames_per_window)
            if 'challenge' not in self.splits:
                num_batches = len(players)
                self._load_labels(match_path, half, num_batches)
            self.players.extend(players)

        if 'challenge' not in self.splits:
            self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        players = self.players[index]
        labels = torch.from_numpy(self.labels[index, :]) if 'challenge' not in self.splits else None
        return players, labels

    def __len__(self):
        return len(self.players)

    def collate(self):
        def _collate_challenge(examples: List):
            return Batch.from_data_list([x for b in examples for x in b[0]])

        def _collate(examples: List):
            return Batch.from_data_list([x for b in examples for x in b[0]]), \
                   torch.stack([x[1] for x in examples], dim=0)

        return _collate_challenge if 'challenge' in self.splits else _collate


class SoccerNetGCNTesting(SoccerNetGraphBase):
    def __init__(self, data_dir: Path, splits: List[str], graph_stream: GraphStream, window_size_sec=60, frame_rate=2,
                 distance_players=25, invert_second_half_positions=True, filter_positions=True, min_detection_score=0,
                 use_velocity=True, use_player_prediction=True, calibration_flags=None, tiny=None, cache=True,
                 overwrite_cache=False):
        super().__init__(data_dir, splits, graph_stream, window_size_sec=window_size_sec, frame_rate=frame_rate,
                         distance_players=distance_players, invert_second_half_positions=invert_second_half_positions,
                         filter_positions=filter_positions, min_detection_score=min_detection_score,
                         use_velocity=use_velocity, use_player_prediction=use_player_prediction,
                         calibration_flags=calibration_flags)

        if tiny:
            assert tiny % 2 == 0, "tiny variable must be an even integer"

        self.match_graphs = self.load_graphs(cache, overwrite_cache, tiny)

        self.half_matches = self.half_matches[:tiny]
        self.matches = [hm[0] for hm in self.half_matches[::2]]

    def _load_graph_item(self, match_path, half, stride, off=0):
        features = self.match_graphs[match_path][half]['features']
        edges = self.match_graphs[match_path][half]['edges']
        return self._half_match_to_graph(features, edges, stride, off)

    def __getitem__(self, index):
        match_path = self.matches[index]
        players = [self._load_graph_item(match_path, half, 1, self.frames_per_window // 2) for half in range(2)]
        return str(match_path), players[0], players[1]

    def __len__(self):
        return len(self.matches)

    @staticmethod
    def collate(examples: List):
        return [x[0] for x in examples], \
               Batch.from_data_list([c for b in examples for x in b[1] for c in x]), \
               Batch.from_data_list([c for b in examples for x in b[2] for c in x])
