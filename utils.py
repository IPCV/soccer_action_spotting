import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import yaml
from addict import Dict

from IO.datasets import GraphStream
from IO.datasets import FeatureStream
import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def parse_model_conf(json_fpath: Path, weights_dir: Path):
    with json_fpath.open() as json_file:
        conf = json.load(json_file, object_pairs_hook=OrderedDict)

        # The order affects the model architecture and the data feeding
        streams = []
        for s, v in conf['model']['streams'].items():
            if s == 'graph' and conf['model'].get('graph', None):
                backbone_name = conf['model']['graph']['backbone']['name']

                use_player_prediction = conf['model']['graph'].get('use_player_prediction', True)
                conf['model']['graph']['use_player_prediction'] = use_player_prediction

                use_velocity = conf['model']['graph'].get('use_velocity', True)
                conf['model']['graph']['use_velocity'] = use_velocity

                graph_stream = GraphStream(f'{s}:{backbone_name}',
                                           v['bboxes'], v['calibrations'],
                                           use_player_prediction,
                                           use_velocity)
                streams.append(graph_stream)
                conf['model']['graph']['feature_vector_size'] = graph_stream.feature_vector_size
            else:
                streams.append(FeatureStream(s, v['features']))
        conf = Dict(conf)
        conf.model.streams = streams

    conf.model.has_graphs = any([s.name.startswith('graph') for s in streams])
    conf.model.num_streams = len(streams)

    conf.model.identifier = f'{conf.model.pool_layer.name}_' + '_'.join(s.name for s in streams)
    conf.model.weights_dir = weights_dir.joinpath(conf.model.identifier)
    return conf


def parse_logs(args, model_conf):
    log_fname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    log_fpath = model_conf.weights_dir.joinpath('logs', log_fname)
    with open(args.log_config, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = str(log_fpath)

    return Dict({'log_config': log_config,
                 'log_fpath': log_fpath})
