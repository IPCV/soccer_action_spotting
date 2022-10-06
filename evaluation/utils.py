import json
import os
import zipfile
from itertools import product


def to_game_time(half, frame_index, frame_rate):
    seconds = int((frame_index // frame_rate) % 60)
    minutes = int((frame_index // frame_rate) // 60)
    return f'{half + 1} - {minutes}:{seconds}'


def to_json_pred(dataset, half, label, frame_index, confidence):
    return {'gameTime': to_game_time(half, frame_index, dataset.frame_rate),
            'label': dataset.classes[label + 1],
            'position': str(int((frame_index / dataset.frame_rate) * 1000)),
            'half': str(half + 1),
            'confidence': str(confidence)}


def make_json_results(dataset, results_dir, match_id, timestamps, get_spots):
    num_classes, halves = dataset.num_classes - 1, [0, 1]
    json_data = {'UrlLocal': match_id,
                 'predictions': []}
    for (h, t), l in product(zip(halves, timestamps), range(num_classes)):
        json_data["predictions"].extend([to_json_pred(dataset, h, l, int(s[0]), s[1]) for s in get_spots(t[:, l])])

    game_dir = results_dir.joinpath(match_id)
    game_dir.mkdir(parents=True, exist_ok=True)

    with open(game_dir.joinpath("results_spotting.json"), 'w') as results_file:
        json.dump(json_data, results_file, indent=4)


def zip_results(zip_path, target_dir, filename="results_spotting.json"):
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(str(target_dir)) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])
