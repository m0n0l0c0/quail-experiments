import os
import sys
import pickle
import numpy as np

from tqdm import tqdm
from shutil import rmtree, copyfile
from pathlib import Path
from functools import reduce
from collections import Counter
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

# separate normalization
NORM_FEATS = [["embeddings", "logits"], ["contexts", "question", "endings"]]
DEFAULT_FEATS = [
    "embeddings", "logits", "contexts", "question", "endings", "text_length"
]


def flatten(array):
    ret = []
    if not isinstance(array, list):
        return array
    for item in array:
        if isinstance(item, list):
            ret.extend(flatten(item))
        else:
            ret.append(item)
    return ret


def get_unique_features(features):
    return list(Counter(flatten(features)).keys())


def get_data_size_gb(feature_dict):
    n_elems = [list(feat.shape) for feat in feature_dict.values()]
    gb = 1024 ** 3
    float_size = sys.getsizeof(np.float32())
    total_size_gb = sum([
        ((float_size * e) / gb) for e in flatten(n_elems)
    ])
    return total_size_gb


def get_item_from_data(features, idx):
    from dataset_class import Dataset
    if isinstance(features, Dataset):
        return features._get_x(idx, collate=False, unshape=True)
    else:
        return {
            f: np.array(features[f][idx]) for f in features.keys()
        }


def save_data(output_dir, prefix, single_items=False, **kwargs):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if single_items:
        for key, value in kwargs.items():
            fpath = os.path.join(output_dir, f"{prefix}_{key}.pkl")
            with open(fpath, "wb") as fout:
                pickle.dump(value, fout)

    fpath = os.path.join(output_dir, f"{prefix}_data.pkl")
    with open(fpath, "wb") as fout:
        pickle.dump(kwargs, fout)


def load_dataset(data_path):
    data = pickle.load(open(data_path, "rb"))
    return data


def setup_data_dir(data_dir, overwrite):
    if data_dir is None:
        raise ValueError(
            "A 'data_dir' must be provided to save the dataset!"
        )
    data_dir = Path(data_dir)
    if (
        (overwrite and data_dir.exists()) or
        (data_dir.exists() and len(list(data_dir.iterdir())) == 0)
    ):
        rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=overwrite)
    return data_dir


def copy_file(src_dst):
    src, dst = src_dst
    copyfile(src, dst)
    return True


def nelems_from_shape(shape):
    return reduce(lambda a, b: a * b, shape)


def parallel_copy(files):
    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        with tqdm(desc="Copy files", total=len(files)) as progress:
            futures = []
            for src_dst in files:
                future = pool.submit(copy_file, src_dst)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)

            progress.close()
            return results
