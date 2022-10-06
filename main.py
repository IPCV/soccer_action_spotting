import json
import logging.config
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from addict import Dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from IO.datasets import SoccerNetFeatures
from IO.datasets import SoccerNetFeaturesTesting
from IO.datasets import SoccerNetGCN
from IO.datasets import SoccerNetGCNTesting
from evaluation import evaluate
from loss import NLLLoss
from models import ActionSpotter
from models import GraphActionSpotter
from models import Multistream
from train import trainer
from utils import parse_model_conf, parse_logs, get_git_revision_hash


def load_model(model_args, train_args):

    parameters = {'pool': model_args.pool_layer.name,
                  'window_size_sec': model_args.window_size_sec,
                  'frame_rate': model_args.frame_rate,
                  'num_classes': model_args.num_classes,
                  'vocab_size': model_args.pool_layer.vocab_size,
                  'weights': train_args.weights}

    if model_args.has_graphs:
        parameters.update({'gcn_backbone': model_args.graph.backbone.name,
                           'feature_multiplier': model_args.graph.backbone.feature_multiplier,
                           'use_calibration_confidence': model_args.graph.calibration_flags.use_confidence,
                           'feature_vector_size': model_args.graph.feature_vector_size})

    if model_args.num_streams > 1:
        return Multistream(streams=model_args.streams, **parameters)

    if model_args.has_graphs:
        return GraphActionSpotter(**parameters)

    return ActionSpotter(input_size=model_args.streams[0].features.dim, **parameters)


def get_dataset_parameters(model_args, train_args, main_args):

    parameters = {'data_dir': train_args.dataset_path,
                  'window_size_sec': model_args.window_size_sec,
                  'frame_rate': model_args.frame_rate,
                  'tiny': main_args.tiny}

    if model_args.has_graphs and model_args.graph:
        parameters.update({'distance_players': model_args.graph.distance_players,
                           'invert_second_half_positions': model_args.graph.invert_second_half_positions,
                           'filter_positions': model_args.graph.filter_positions,
                           'min_detection_score': model_args.graph.min_detection_score,
                           'use_velocity': model_args.graph.use_velocity,
                           'use_player_prediction': model_args.graph.use_player_prediction,
                           'calibration_flags': model_args.graph.calibration_flags,
                           'cache': main_args.cache,
                           'overwrite_cache': main_args.overwrite_cache})

    if model_args.has_graphs and model_args.graph:
        if model_args.num_streams == 1:
            parameters.update({'graph_stream': model_args.streams[0]})
        else:
            parameters.update({'streams': model_args.streams})
    else:
        parameters.update({'feature_streams': model_args.streams})
    return parameters


def main(model_args, opt_args, train_args, main_args):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(opt_args)
    logging.info(train_args)

    model = load_model(model_args, train_args).cuda()
    logging.info(model)

    total_args = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_args))

    dataset_parameters = get_dataset_parameters(model_args, train_args, main_args)

    if not main_args.test_only:
        if model_args.has_graphs and model_args.num_streams == 1:
            logging.info("Loading training split")
            training_dataset = SoccerNetGCN(splits=train_args.splits.train, **dataset_parameters)
            training_collate = training_dataset.collate()

            logging.info("Loading validation split")
            validation_dataset = SoccerNetGCN(splits=train_args.splits.valid, **dataset_parameters)
            validation_collate = validation_dataset.collate()
        else:
            logging.info("Loading training split")
            training_dataset = SoccerNetFeatures(splits=train_args.splits.train, **dataset_parameters)
            training_collate = None

            logging.info("Loading validation split")
            validation_dataset = SoccerNetFeatures(splits=train_args.splits.valid, **dataset_parameters)
            validation_collate = None

        training_loader = DataLoader(training_dataset,
                                     batch_size=opt_args.batch_size,
                                     shuffle=True,
                                     num_workers=train_args.max_num_workers,
                                     pin_memory=True,
                                     collate_fn=training_collate)

        validation_loader = DataLoader(validation_dataset,
                                       batch_size=opt_args.batch_size,
                                       shuffle=False,
                                       num_workers=train_args.max_num_workers,
                                       pin_memory=True,
                                       collate_fn=validation_collate)

        criterion = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt_args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0,
                                     amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                               patience=opt_args.patience)
        writer = SummaryWriter(train_args.log_dir, train_args.comment)

        Loaders = namedtuple('Loaders', 'train valid')
        trainer(Loaders(training_loader, validation_loader), model, optimizer, scheduler,
                criterion, writer, model_args.weights_dir, opt_args.max_epochs, train_args.evaluation_frequency)
        writer.close()
        del training_loader, training_dataset
        del validation_loader, validation_dataset

    checkpoint = torch.load(model_args.weights_dir.joinpath("model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # evaluate on multiple splits [test/challenge]
    for split in train_args.splits.test:

        if model_args.num_streams == 1 and model_args.has_graphs:
            test_dataset = SoccerNetGCNTesting(splits=[split], **dataset_parameters)
            test_collate = SoccerNetGCNTesting.collate
        else:
            test_dataset = SoccerNetFeaturesTesting(splits=[split], **dataset_parameters)
            test_collate = None

        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 collate_fn=test_collate)

        results = evaluate(test_loader, model,
                           model_args.weights_dir,
                           model_args.nms.window_sec,
                           model_args.nms.threshold,
                           overwrite=main_args.overwrite_test_results,
                           tiny=main_args.tiny,
                           test_batch_size=main_args.test_batch_size)
        if results is None:
            continue

        logging.info('Best Performance at end of training ')
        logging.info(f'a_mAP visibility all: {results["a_mAP"]}')
        logging.info(f'a_mAP visibility all per class: {results["a_mAP_per_class"]}')
        logging.info(f'a_mAP visibility visible: {results["a_mAP_visible"]}')
        logging.info(f'a_mAP visibility visible per class: {results["a_mAP_per_class_visible"]}')
        logging.info(f'a_mAP visibility not shown: {results["a_mAP_unshown"]}')
        logging.info(f'a_mAP visibility not shown per class: {results["a_mAP_per_class_unshown"]}')


def parse_args():
    parser = ArgumentParser(description='Multi-modal action spotting implementation')
    parser.add_argument('conf', help='JSON model configuration filepath',
                        type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--split_train', nargs='+', dest='train_splits',
                        help='list of splits for training (default: [train])',
                        default=["train"])
    parser.add_argument('--split_valid', nargs='+', dest='validation_splits',
                        help='list of splits for validation (default: [valid])',
                        default=["valid"])
    parser.add_argument('--split_test', nargs='+', dest='test_splits',
                        help='list of split for testing (default: [test])',
                        default=["test"])
    parser.add_argument('--max_num_workers', required=False,
                        help='number of worker to load data (default: 4)',
                        default=4, type=int)
    parser.add_argument('--evaluation_frequency', required=False,
                        help='Evaluation frequency in number of epochs (default: 10)',
                        default=10, type=int)
    parser.add_argument('--weights_dir', required=False,
                        help='Path for weights saving directory (default: weights)',
                        default="weights", type=lambda p: Path(p))
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('--GPU', required=False,
                        help='ID of the GPU to use (default: 0)',
                        default=0, type=int)
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    parser.add_argument('--test_only', required=False,
                        help='Perform testing only (default: False)',
                        action='store_true')
    parser.add_argument('--test_batch_size', required=False,
                        help='Batch size for testing (default: 256)',
                        default=256, type=int)
    parser.add_argument('--overwrite_test_results', required=False,
                        help='Overwrite test results (default: True)',
                        action='store_false')
    parser.add_argument('--tiny', required=False,
                        help='Consider smaller amount of games (default: None)',
                        type=int, default=None)
    parser.add_argument('--num_random_samples', required=False,
                        help='Consider smaller random amount of games (default: None)',
                        type=int, default=None)
    parser.add_argument('--no_cache', required=False,
                        help="Don't use cached data (default: False)",
                        action='store_true')
    parser.add_argument('--overwrite_cache', required=False,
                        help='Overwrite cached data (default: False)',
                        action='store_true')

    args = parser.parse_args()
    conf = parse_model_conf(args.conf, args.weights_dir)

    comment_tmp = f'vocab_size:{0} win_size:{1} frame_rate:{2} lr:{3} batch_size:{4}'
    comment = comment_tmp.format(conf.model.pool_layer.vocab_size,
                                 conf.model.window_size_sec,
                                 conf.model.frame_rate,
                                 conf.opt.learning_rate,
                                 conf.opt.batch_size)

    training = Dict({'dataset_path': args.dataset_path,
                     'splits': {'train': args.train_splits,
                                'valid': args.validation_splits,
                                'test': args.test_splits},
                     'max_num_workers': args.max_num_workers,
                     'evaluation_frequency': args.evaluation_frequency,
                     'weights': args.weights,
                     'log_dir': conf.model.weights_dir.joinpath('runs', comment),
                     'comment': comment})

    main_args = Dict({'test_only': args.test_only,
                      'test_batch_size': args.test_batch_size,
                      'tiny': args.tiny,
                      'num_random_samples': args.num_random_samples,
                      'cache': not args.no_cache,
                      'overwrite_cache': args.overwrite_cache,
                      'overwrite_test_results': args.overwrite_test_results})

    logs = parse_logs(args, conf.model)

    return Dict({'model': conf.model,
                 'optimization': conf.optimization,
                 'training': training,
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
    logging.info('Starting main function')
    main(args.model, args.optimization, args.training, args.main)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
