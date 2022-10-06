import os
import time
from functools import partial
from pathlib import Path

import numpy as np
from SoccerNet.Evaluation.ActionSpotting import evaluate as soccer_net_evaluate
from SoccerNet.Evaluation.utils import AverageMeter
from tqdm import tqdm
from torch_geometric.data import Batch

from .utils import make_json_results
from .utils import zip_results
from math import ceil


def get_spot_from_nms(input_, window=60, thresh=0.0):
    detections_tmp = np.copy(input_)
    indexes, max_values = [], []

    while np.max(detections_tmp) >= thresh:
        # Get the max remaining index and value
        max_value, max_index = np.max(detections_tmp), np.argmax(detections_tmp)
        max_values.append(max_value)
        indexes.append(max_index)

        nms_from = int(np.maximum(max_index - (window / 2), 0))
        nms_to = int(np.minimum(max_index + int(window / 2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1
    return np.transpose([indexes, max_values])


def compute_ouput_batch(model, features, batch_size=256):

    if isinstance(features, Batch):
        features = features.to_data_list()
        num_features = len(features)

        batch_size *= model.frames_per_window

        timestamp_long = []
        for b in range(ceil(num_features / batch_size)):
            start_frame = batch_size * b
            end_frame = min(num_features, start_frame + batch_size)

            f = Batch.from_data_list(features[start_frame:end_frame])
            output = model(f.cuda())
            output = output.cpu().detach().numpy()

            timestamp_long.append(output)
        timestamp_long = np.concatenate(timestamp_long)
    else:
        features = [f.squeeze(0) for f in features]

        timestamp_long = []
        num_features = len(features[0])
        for b in range(ceil(num_features / batch_size)):
            start_frame = batch_size * b
            end_frame = min(num_features, start_frame + batch_size)

            f = [f[start_frame:end_frame].cuda() for f in features]
            output = model(f[0]) if len(features) == 1 else model(f)
            output = output.cpu().detach().numpy()

            timestamp_long.append(output)
        timestamp_long = np.concatenate(timestamp_long)

    # Removing Background category
    timestamp_long = timestamp_long[:, 1:]

    return timestamp_long


def test(model, loader, nms_window, nms_threshold, batch_size, results_dir):
    batch_time, data_time = AverageMeter(), AverageMeter()
    description = 'Test (spot): ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) '

    get_spots = partial(get_spot_from_nms,
                        window=nms_window * loader.dataset.frame_rate,
                        thresh=nms_threshold)

    model.eval()

    end = time.time()
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (match_id, feat_half1, feat_half2) in t:
            data_time.update(time.time() - end)

            # Batch size of 1
            match_id = match_id[0]
            predictions = [compute_ouput_batch(model, f, batch_size) for f in [feat_half1, feat_half2]]

            batch_time.update(time.time() - end)
            end = time.time()

            t.set_description(description.format(avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val))
            make_json_results(loader.dataset, results_dir, match_id, predictions, get_spots)


def evaluate(loader, model, target_dir: Path, nms_window=30, nms_threshold=0.5, overwrite=True, tiny=None, test_batch_size=256):
    splits = '_'.join(loader.dataset.splits)
    zipped_results = target_dir.joinpath(f"results_spotting_{splits}.zip")

    if not os.path.exists(zipped_results) or overwrite:
        results_dir = target_dir.joinpath(f"outputs_{splits}")

        test(model, loader, nms_window, nms_threshold, test_batch_size, results_dir)
        zip_results(zipped_results, results_dir)

    if 'challenge' in splits:
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None

    if not tiny:
        return None

    version = 2 if loader.dataset.num_classes > 4 else 1
    return soccer_net_evaluate(SoccerNet_path=str(loader.dataset.path),
                               Predictions_path=str(zipped_results),
                               prediction_file="results_spotting.json",
                               split="test",
                               version=version)
