import json
import logging.config
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
from math import ceil
from pathlib import Path

import numpy as np
import torch
from addict import Dict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from IO.datasets import SoccerNetFeaturesTesting
from IO.datasets import SoccerNetGCNTesting
from models import ActionSpotter
from models import GraphActionSpotter
from models import Multistream
from utils import parse_model_conf, parse_logs, get_git_revision_hash


class Hook():
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def compute_layer_output(model, layer, features):
    hook = Hook(layer)

    if isinstance(features, Batch):
        features = features.to_data_list()
        num_features = len(features)

        batch_size = model.frames_per_window

        layer_outputs = []
        for b in range(ceil(num_features / batch_size)):
            start_frame = batch_size * b
            end_frame = min(num_features, start_frame + batch_size)

            f = Batch.from_data_list(features[start_frame:end_frame])
            model(f.cuda())
            layer_outputs.append(hook.output.cpu().detach().numpy())
        layer_outputs = np.concatenate(layer_outputs)
    else:
        features = [f.squeeze(0) for f in features]
        num_features = len(features[0])

        layer_outputs = []
        for b in range(num_features):
            start_frame = b
            end_frame = min(num_features, start_frame + 1)

            f = [f[start_frame:end_frame].cuda() for f in features]
            model(f[0]) if len(features) == 1 else model(f)
            layer_outputs.append(hook.output.cpu().detach().numpy())
        layer_outputs = np.concatenate(layer_outputs)

    return layer_outputs


def main(model_args, extraction_args, main_args):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(extraction_args)

    has_graphs = any([s.name.startswith('graph') for s in model_args.streams])
    if len(model_args.streams) == 1:
        if has_graphs:
            model = GraphActionSpotter(model_args.pool_layer.name,
                                       model_args.graph.backbone.name,
                                       model_args.window_size_sec,
                                       model_args.frame_rate,
                                       model_args.num_classes,
                                       model_args.pool_layer.vocab_size,
                                       model_args.graph.backbone.feature_multiplier,
                                       model_args.graph.calibration_flags.use_confidence,
                                       model_args.graph.feature_vector_size,
                                       extraction_args.weights).cuda()
        else:
            model = ActionSpotter(model_args.pool_layer.name,
                                  model_args.streams[0].features.dim,
                                  model_args.window_size_sec,
                                  model_args.frame_rate,
                                  model_args.num_classes,
                                  model_args.pool_layer.vocab_size,
                                  extraction_args.weights).cuda()
    else:
        assert not has_graphs, "Graphs for multiple streams processing not yet implemented"
        model = Multistream([(s.name, s.features.dim) for s in model_args.streams],
                            model_args.pool_layer.name,
                            model_args.window_size_sec,
                            model_args.frame_rate,
                            model_args.num_classes,
                            model_args.pool_layer.vocab_size,
                            extraction_args.weights).cuda()
    logging.info(model)
    assert hasattr(model, extraction_args.layer), f"Model has not layer named {extraction_args.layer}"

    for split in extraction_args.splits:
        if has_graphs:
            dataset = SoccerNetGCNTesting(extraction_args.dataset_path,
                                          [split],
                                          model_args.streams[0],
                                          model_args.window_size_sec,
                                          model_args.frame_rate,
                                          model_args.graph.distance_players,
                                          model_args.graph.invert_second_half_positions,
                                          model_args.graph.filter_positions,
                                          model_args.graph.min_detection_score,
                                          model_args.graph.use_velocity,
                                          model_args.graph.use_player_prediction,
                                          model_args.graph.calibration_flags,
                                          main_args.tiny,
                                          main_args.cache,
                                          main_args.overwrite_cache)

            test_collate = SoccerNetGCNTesting.collate
        else:
            dataset = SoccerNetFeaturesTesting(extraction_args.dataset_path,
                                               [split],
                                               model_args.streams,
                                               model_args.window_size_sec,
                                               model_args.frame_rate,
                                               main_args.tiny)
            test_collate = None

        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=extraction_args.max_num_workers,
                            pin_memory=True,
                            collate_fn=test_collate)

        model.eval()
        extraction_layer = getattr(model, extraction_args.layer)

        with tqdm(enumerate(loader), total=len(loader)) as t:
            for i, (match_id, feat_half1, feat_half2) in t:
                # Batch size of 1
                match_id = match_id[0]

                for j, feat_half in enumerate([feat_half1, feat_half2]):
                    features_filename = extraction_args.features_filename.format(extraction_layer.out_features, j+1)
                    extraction_fpath = extraction_args.dataset_path.joinpath(match_id, features_filename)
                    try:
                        if not main_args.overwrite_features and extraction_fpath.exists():
                            continue
                        features = feat_half.to_data_list()
                        outputs = compute_layer_output(model, extraction_layer, feat_half)
                        np.save(extraction_fpath, outputs)
                    except Exception as e:
                        logging.info(f'An exception occurred: {e}')


def parse_args():
    parser = ArgumentParser(description='Multi-modal action spotting implementation')
    parser.add_argument('layer', help='Layer name from the model to extract features from (default: None)',
                        default=None, type=str)
    parser.add_argument('conf', help='JSON model configuration filepath',
                        type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--splits', nargs='+', dest='splits',
                        help='list of splits for training (default: [train])',
                        default=["train", "valid", "test"])
    parser.add_argument('--max_num_workers', required=False,
                        help='number of worker to load data (default: 1)',
                        default=1, type=int)
    parser.add_argument('--weights_dir', required=False,
                        help='Path for weights saving directory (default: weights)',
                        default="weights", type=lambda p: Path(p))
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('--GPU', required=False,
                        help='ID of the GPU to use (default: -1)',
                        default=-1, type=int)
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    parser.add_argument('--tiny', required=False,
                        help='Consider smaller amount of games (default: None)',
                        type=int, default=None)
    parser.add_argument('--no_cache', required=False,
                        help="Don't use cached data (default: False)",
                        action='store_true')
    parser.add_argument('--overwrite_features', required=False,
                        help='Overwrite features cached data (default: False)',
                        action='store_true')

    args = parser.parse_args()
    conf = parse_model_conf(args.conf, args.weights_dir)

    if args.weights is None:
        weights = conf.model.weights_dir.joinpath("model.pth.tar")
    else:
        weights = args.weights

    extraction = Dict({'dataset_path': args.dataset_path,
                       'layer': args.layer,
                       'features_filename': f'{conf.model.identifier} layer:{args.layer} vector_size:' + '{} half:{}.npy',
                       'splits': args.splits,
                       'max_num_workers': args.max_num_workers,
                       'weights': weights})

    main_args = Dict({'tiny': args.tiny,
                      'overwrite_features': args.overwrite_features,
                      'cache': not args.no_cache,
                      })

    logs = parse_logs(args, conf.model)

    return Dict({'model': conf.model,
                 'optimization': conf.optimization,
                 'extraction': extraction,
                 'logs': logs,
                 'GPU': args.GPU,
                 'main': main_args,
                 'program_args': args})


if __name__ == '__main__':

    args = parse_args()

    args.logs.log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(args.logs.log_config)

    logging.info("Input arguments:")
    logging.info(args.program_args)

    with args.program_args.conf.open() as json_file:
        conf = json.load(json_file, object_pairs_hook=OrderedDict)

    logging.info("Configuration:")
    logging.info(conf)

    logging.info(f"Git commit hash: {get_git_revision_hash()}")

    torch.manual_seed(args.optimization.random_seed)
    np.random.seed(args.optimization.random_seed)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info('Starting layer output extraction function')
    main(args.model, args.extraction, args.main)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
