import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd

from shutil import rmtree
from pathlib import Path
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split

# separate normalization
NORM_FEATS = [["embeddings", "logits"], ["contexts", "question", "endings"]]
DEFAULT_FEATS = [["embeddings", "logits", "contexts", "question", "endings"]]


def get_flat_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def get_unique_features(features):
    return list(Counter(get_flat_list(features)).keys())


def get_data_size_gb(feature_dict):
    n_elems = [list(feat.shape) for feat in feature_dict.values()]
    gb = 1024 ** 3
    float_size = sys.getsizeof(np.float32())
    total_size_gb = sum([
        ((float_size * e) / gb) for e in get_flat_list(n_elems)
    ])
    return total_size_gb


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


class DatasetOperation(object):
    def __init__(self, name, feature_set, fn, call_by_features=True):
        self.name = name
        self.call_by_features = call_by_features
        self.feature_set = feature_set
        if feature_set != "all":
            self.feature_set = get_unique_features(feature_set)
        self.fn = fn

    def __call__(self, data):
        if not self.call_by_features:
            data = self.fn(data)
        else:
            feature_set = self.feature_set
            if self.feature_set == "all":
                feature_set = list(data.keys())
            for feature in feature_set:
                data[feature] = self.fn(data[feature])
        return data

    def fn(self, data):
        return data


class ReshapeOp(DatasetOperation):
    def __init__(self, feature_set=None):
        if feature_set is None:
            feature_set = "all"
        super(ReshapeOp, self).__init__(
            name="ReshapeOp",
            feature_set=feature_set,
            fn=self.reshape,
        )

    def reshape(self, data):
        data = data.reshape(1, *data.shape)
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        elif len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)
        return data


class ConcatOp(DatasetOperation):
    def __init__(self, feature_set=None):
        if feature_set is None:
            feature_set = "all"
        super(ConcatOp, self).__init__(
            name="ConcatOp",
            feature_set=feature_set,
            fn=self.concat,
            call_by_features=False,
        )

    def concat(self, data):
        ret = None
        for feat in data.keys():
            if ret is None:
                ret = np.array(data[feat])
            else:
                ret = np.concatenate([ret, data[feat]], axis=1)
        return ret


class NormalizeFeaturesOp(DatasetOperation):
    def __init__(self, features):
        super(NormalizeFeaturesOp, self).__init__(
            name="normalize",
            feature_set=get_unique_features(features),
            fn=self.normalize_features,
        )

    def __call__(self, data):
        features = self.feature_set
        if self.feature_set == "all":
            features = list(data.keys())
        for norm_group in NORM_FEATS:
            to_norm = [f for f in norm_group if f in features]
            if len(to_norm) > 0:
                data = self.normalize_features(to_norm, data)

        return data

    def normalize_features(self, features, data):
        orig_shapes = [data[f].shape for f in features]
        reshape_op = ReshapeOp(feature_set=features)
        concat_op = ConcatOp(feature_set=features)
        reshaped_data = reshape_op(data)
        reshapes = [reshaped_data[f].shape for f in features]
        concat_data = concat_op(reshaped_data)
        concat_data = (concat_data.T - concat_data.mean(axis=1)).T
        concat_data = (concat_data.T / concat_data.std()).T
        for o_shape, r_shape, feat in zip(orig_shapes, reshapes, features):
            data[feat] = concat_data[:, :r_shape[-1]]
            data[feat] = data[feat].reshape(o_shape)

        return data


class Dataset(object):
    def __init__(
        self,
        data_path,
        data_frame=None,
        features=None,
        ops=None,
        cast=None,
        ret_x=True,
        ret_y=False,
    ):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.data_frame = data_frame
        self._index_name = "index.csv"
        if data_frame is None:
            index_file = os.path.join(self.data_path, self._index_name)
            self.data_frame = pd.read_csv(index_file)
        self.cast = cast
        self.features = features
        self._ret_x = ret_x
        self._ret_y = ret_y
        self._ret_xy = self._ret_x and self._ret_y
        self.ops = self._setup_ops(ops)
        self.collator = ConcatOp(feature_set=self.features)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X, y, ret = None, None, None
        if self._ret_x:
            X = self._get_x(idx)

        if self._ret_y:
            y = self._get_y(idx)

        if self._ret_xy:
            ret = (X, y)
        elif self._ret_x:
            ret = X
        else:
            ret = y

        return ret

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            res = self[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def _setup_ops(self, ops):
        ret_ops = OrderedDict()
        if ops is None:
            ops = OrderedDict()
        if "cast" in ops:
            ret_ops.update(cast=ops["cast"])
        elif self.cast is not None:
            ret_ops = ret_ops.update(cast=lambda d: d.astype(self.cast))
        if "reshape" in ops:
            ret_ops.update(reshape=ops["reshape"])
        else:
            ret_ops.update(reshape=ReshapeOp())
        for op_key, op_value in ops.items():
            if op_key not in ret_ops:
                ret_ops.update(**{op_key: op_value})

        return ret_ops

    def _get_features_from_data(self, data, **kwargs):
        orig_shapes = None
        if kwargs.get("unshape", False):
            orig_shapes = [data[f].shape for f in data.keys()]
        for op_name, op in self.ops.items():
            data = op(data)
        if "collate" not in kwargs or kwargs["collate"]:
            data = self.collator(data)
        elif orig_shapes is not None:
            # cant collate and unshape
            for shape, feat in zip(orig_shapes, data.keys()):
                data[feat] = data[feat].reshape(shape)
        return data

    def _get_x(self, idx, **kwargs):
        idx = self._iter_idx_to_df_idx(idx)
        emb_data = pickle.load(open(self.data_frame.X[idx], "rb"))
        if isinstance(emb_data, dict):
            emb_data = {k: np.array(v) for k, v in emb_data.items()}
        elif isinstance(emb_data, list):
            emb_data = np.array(emb_data)
        # ToDo := Aggregate features from data_frame
        emb_data = self._get_features_from_data(emb_data, **kwargs)
        return emb_data

    def _get_y(self, idx):
        idx = self._iter_idx_to_df_idx(idx)
        y = self.data_frame.y[idx]
        if self.cast:
            # ToDo := Is this a numpy array?
            y = y.astype(self.cast)
        return y

    def _iter_idx_to_df_idx(self, idx):
        return self.data_frame.index[idx]

    @property
    def ret_x(self):
        return self._ret_x

    @ret_x.setter
    def ret_x(self, value):
        self._ret_x = value
        self._ret_xy = self._ret_x and self.ret_y
    
    @property
    def ret_y(self):
        return self._ret_y

    @ret_y.setter
    def ret_y(self, value):
        self._ret_y = value
        self._ret_xy = self._ret_x and self.ret_y

    def get_x_y_from_dict(self, features=None):
        X_dataset = Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=features,
            ops=self.ops,
            cast=self.cast,
            ret_x=True,
            ret_y=False,
        )
        y_dataset = Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=features,
            ops=self.ops,
            cast=self.cast,
            ret_x=False,
            ret_y=True,
        )
        return X_dataset, y_dataset

    def get_splits(self, test_size=0.25):
        train_df, test_df = train_test_split(
            self.data_frame, test_size=test_size
        )
        train_dataset = Dataset(
            data_path=self.data_path,
            data_frame=train_df,
            features=self.features,
            ops=self.ops,
            cast=self.cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )
        test_dataset = Dataset(
            data_path=self.data_path,
            data_frame=test_df,
            features=self.features,
            ops=self.ops,
            cast=self.cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )
        return train_dataset, test_dataset

    def get_class_proportions(self):
        props = list(Counter(self.data_frame.y.values).values())
        return round(max(props) / min(props))

    def normalize_dataset_by_features(self, features):
        return Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=self.features,
            ops=OrderedDict(normalize=NormalizeFeaturesOp(features)),
            cast=self.cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )

    def save(self, output_dir, overwrite=False, **kwargs):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=overwrite)
        for idx, path in enumerate(self.data_frame.X):
            data_name = path.split("/")[-1]
            X = self._get_x(idx, collate=False, unshape=True)
            pickle.dump(X, open(output_path/data_name, "wb"))
        self.data_frame.to_csv(output_path/self._index_name)

    def add_features(
        self,
        feature_dict,
        in_file=True,
        in_place=True,
        data_dir=None,
        overwrite=False,
    ):
        # ToDo := Pass on this, gathering features could be a pain
        # if in_file == False:
        #     total_size_gb = get_data_size_gb(feature_dict)
        #     if total_size_gb > 1:
        #         print("Warning, adding feature to disk will be "
        #             f"{total_size_gb} in GB, pass 'in_file=True' to save in "
        #             "data_frame"
        #         )
        #     self._add_features_to_index(feature_dict)
        # else:
        def get_item(features, idx):
            if isinstance(features, Dataset):
                return dataset._get_x(idx, collate=False, unshape=True)
            else:
                return {
                    f: np.array(feature_dict[f][idx]) for f in feature_dict.keys()
                }

        if not in_place:
            if data_dir is None:
                raise ValueError(
                    "A 'data_dir' must be provided to save the dataset with"
                    " new features when 'in_place=False'!"
                )
            data_dir = Path(data_dir)
            if (
                (overwrite and data_dir.exists()) or
                (data_dir.exists() and len(list(data_dir.iterdir())) == 0)
            ):
                rmtree(data_dir)

            Path(data_dir).mkdir(parents=True, exist_ok=overwrite)
            _X = []
            _y = []

        if isinstance(feature_dict, Dataset):
            # check dimensions match
            assert(len(feature_dict) == len(self))
        else:
            # check dimensions match
            for feature in feature_dict.values():
                assert(feature.shape[0] == len(self))

        for idx, path in enumerate(self.data_frame.X):
            X = self._get_x(idx, collate=False, unshape=True)
            features = get_item(feature_dict, idx)
            X.update(features)
            if not in_place:
                out_fname = os.path.basename(path)
                out_path = os.path.join(data_dir, out_fname)
                _X.append(out_path)
                _y.append(self._get_y(idx))
            else:
                out_path = path
            pickle.dump(X, open(out_path, "wb"))

        if not in_place:
            pd.DataFrame.from_dict(dict(X=_X, y=_y)).to_csv(
                os.path.join(data_dir, "index.csv")
            )

    def list_features(self):
        first_sample = self._get_x(0, collate=False, unshape=True)
        return list(first_sample.keys())
