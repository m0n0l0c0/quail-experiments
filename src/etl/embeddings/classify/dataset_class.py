import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset as TorchDataset

from dataset_utils import (
    NORM_FEATS,
    get_unique_features,
    nelems_from_shape,
    parallel_copy,
    setup_data_dir,
    get_item_from_data,
)


# Operations on Dataset
class DatasetOperation(object):
    def __init__(self, name, feature_set, fn, call_by_features=True):
        self.name = name
        self.call_by_features = call_by_features

        if feature_set is None:
            feature_set = "all"

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


class CastOp(DatasetOperation):
    def __init__(self, cast, feature_set=None):
        super(CastOp, self).__init__(
            name="CastOp",
            feature_set=feature_set,
            fn=self.cast
        )
        self._cast = cast

    def cast(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                data[key] = self._cast(data[key])
        else:
            data = self._cast(data)

        return data


class ConcatOp(DatasetOperation):
    def __init__(self, feature_set=None):
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
                ret = np.array(data[feat]).reshape(1, -1)
            else:
                ret = np.concatenate([ret, data[feat].reshape(1, -1)], axis=1)
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
        self.features = features

        self._index_name = "index.csv"
        self._index_file = os.path.join(self.data_path, self._index_name)
        if data_frame is None:
            self.data_frame = pd.read_csv(self._index_file)
        self._cast = cast
        self._ret_x = ret_x
        self._ret_y = ret_y
        self._ret_xy = self._ret_x and self._ret_y

        self.ops = self._setup_ops(ops)
        self.post_ops = OrderedDict()
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

        if "cast" in ops or self._cast is not None:
            cast_op = ops["cast"] if "cast" in ops else CastOp(self._cast)
            ret_ops.update(cast=cast_op)

        reshape_op = ops["reshape"] if "reshape" in ops else ReshapeOp()
        ret_ops.update(reshape=reshape_op)

        for op_key, op_value in ops.items():
            if op_key not in ret_ops:
                ret_ops.update(**{op_key: op_value})

        return ret_ops

    def _get_features_from_data(self, data, **kwargs):
        orig_shapes = None
        if kwargs.get("unshape", False):
            # may want to get original features
            orig_shapes = [data[f].shape for f in data.keys()]

        # apply operations
        for op_name, op in self.ops.items():
            data = op(data)

        # collate if necessary
        if "collate" not in kwargs or kwargs["collate"]:
            data = self.collator(data)

        # apply operations after collation (like unsqueezing for NN training)
        for op_name, op in self.post_ops.items():
            data = op(data)

        # reshape original features if requested
        if orig_shapes is not None:
            # cant collate and unshape
            for shape, feat in zip(orig_shapes, data.keys()):
                orig_size = nelems_from_shape(shape)
                new_size = nelems_from_shape(data[feat].shape)
                # op may have reduce data (decomposition, etc)
                if orig_size == new_size:
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
        if self._cast:
            # ToDo := Is this a numpy array?
            y = y.astype(self._cast)
        return y

    def _iter_idx_to_df_idx(self, idx):
        return self.data_frame.index[idx]

    def _prepare_iter_output(self, output, return_dict, batch_size):
        if batch_size == 1:
            return output[0]

        ret = output
        if not return_dict:
            cast = np.concatenate
            if len(np.array(output).shape) == 1:
                cast = np.array
            ret = cast(output)
        else:
            ret = {
                key: np.array([e[key] for e in output])
                for key in output[0].keys()
            }

        return ret

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

    @property
    def cast(self):
        return self._cast

    @cast.setter
    def cast(self, cast):
        if self._cast != cast:
            self._cast = cast
            cast_op = CastOp(self._cast)
            self.ops.update(cast=cast_op)

    @staticmethod
    def build_index(data_dir, y):
        embedding_files = glob.glob(f"{data_dir}/*_data.pkl")
        embedding_files.sort()
        index_path = os.path.join(data_dir, "index.csv")
        index = pd.DataFrame.from_dict(dict(X=embedding_files, y=y))
        index.to_csv(index_path)

    def add_op(self, op_fn, features=None, name=None, after_collate=False):
        if isinstance(op_fn, DatasetOperation):
            operation = op_fn
            if name is None:
                name = op_fn.name
        else:
            if name is None:
                name = f"user_op_{len(self.ops)}"
            operation = DatasetOperation(
                name=name,
                fn=op_fn,
                feature_set=features,
            )

        if after_collate:
            # if operation is inserted after collation,
            # it will not have features, just plain data
            operation.call_by_features = False
            self.post_ops.update(**{name: operation})
        else:
            self.ops.update(**{name: operation})

    def copy_to_dir(self, data_dir, start_idx=0, overwrite=False):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
        files = []
        for idx, path in enumerate(self.data_frame.X.values):
            if start_idx > 0:
                out_fname = str(start_idx + idx)
                out_fname = "0" * (6 - len(out_fname)) + out_fname
                out_fname = f"{out_fname}_data.pkl"
            else:
                out_fname = os.path.basename(path)
            out_path = os.path.join(data_dir, out_fname)
            files.append((path, out_path))

        parallel_copy(files)
        self.data_frame.to_csv(os.path.join(data_dir, "index.csv"))

    def get_x_y_from_dict(self, features=None):
        X_dataset = Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=features,
            ops=self.ops,
            cast=self._cast,
            ret_x=True,
            ret_y=False,
        )
        y_dataset = Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=features,
            ops=self.ops,
            cast=self._cast,
            ret_x=False,
            ret_y=True,
        )
        return X_dataset, y_dataset

    def get_splits(self, test_size=0.25, random_state=None):
        train_df, test_df = train_test_split(
            self.data_frame, test_size=test_size, random_state=None
        )
        train_dataset = Dataset(
            data_path=self.data_path,
            data_frame=train_df,
            features=self.features,
            ops=self.ops,
            cast=self._cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )
        test_dataset = Dataset(
            data_path=self.data_path,
            data_frame=test_df,
            features=self.features,
            ops=self.ops,
            cast=self._cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )
        return train_dataset, test_dataset

    def get_class_proportions(self):
        props = list(Counter(self.data_frame.y.values).values())
        return round(max(props) / min(props))

    def destructure_sample(self, data):
        return self._get_features_from_data(data)

    def iter_features(self, start=None):
        iter_options = dict(collate=False, unshape=True)
        self.n = 0 if start is None else max(min(start, len(self)), 0)
        while self.n < len(self):
            X_dict, y_dict = dict(), dict()
            if self.ret_x:
                X_dict.update(**self._get_x(self.n, **iter_options))
            if self.ret_y:
                y_dict.update(dict(label=self._get_y(self.n)))

            if self._ret_xy:
                ret = (X_dict, y_dict)
            elif self._ret_x:
                ret = X_dict
            else:
                ret = y_dict
            yield ret

            self.n += 1

    def iter(self, return_dict=False, x=None, y=None, batch_size=None):
        if x is not None:
            self.ret_x = x
        if y is not None:
            self.ret_y = y

        if not self.ret_x and not self.ret_y:
            raise ValueError(
                "Must iterate over at least one variable (x/y)"
            )

        if batch_size is None:
            batch_size = 1

        if return_dict:
            iterator = self.iter_features()
        else:
            iterator = iter(self)

        _x = []
        _y = []
        for idx, sample in enumerate(iterator):
            if self._ret_xy:
                _x.append(sample[0])
                _y.append(sample[1])
            else:
                _x.append(sample)
            if (idx + 1) % batch_size == 0 or idx == (len(self) - 1):
                _x = self._prepare_iter_output(
                    _x, return_dict=return_dict, batch_size=batch_size
                )
                if self._ret_xy:
                    _y = self._prepare_iter_output(
                        _y, return_dict=return_dict, batch_size=batch_size
                    )
                    yield _x, _y
                else:
                    yield _x
                _x = []
                _y = []

    def first(self, return_dict=False, x=None, y=None):
        prev_ret_x, prev_ret_y = self.ret_x, self.ret_y

        if x is not None:
            self.ret_x = x
        if y is not None:
            self.ret_y = y

        ret = None
        X_data = None
        y_data = None
        iter_options = {}
        if return_dict:
            iter_options = dict(collate=False, unshape=True)

        if self.ret_x:
            X_data = self._get_x(0, **iter_options)
        if self.ret_y:
            y_data = self._get_y(0)
            if return_dict:
                y_data = dict(label=y_data)

        if self._ret_xy:
            ret = (X_data, y_data)
        elif self.ret_x:
            ret = X_data
        else:
            ret = y_data

        self.ret_x, self.ret_y = prev_ret_x, prev_ret_y
        return ret

    def to_list(self):
        ret = []
        for sample in self.iter(batch_size=20):
            ret.extend(sample)
        return ret

    def normalize_dataset_by_features(self, features):
        return Dataset(
            data_path=self.data_path,
            data_frame=self.data_frame,
            features=self.features,
            ops=OrderedDict(normalize=NormalizeFeaturesOp(features)),
            cast=self._cast,
            ret_x=self._ret_x,
            ret_y=self._ret_y,
        )

    def save(self, output_dir=None, overwrite=False):
        if output_dir is None:
            output_path = Path(self.data_path)
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=overwrite)

        _X = []
        for idx, path in enumerate(self.data_frame.X):
            data_name = path.split("/")[-1]
            X = self._get_x(idx, collate=False, unshape=True)
            output_data_path = output_path/data_name
            pickle.dump(X, open(output_data_path, "wb"))
            _X.append(output_data_path)

        index_data = dict(X=_X, y=self.data_frame.y.to_list())
        index = pd.DataFrame.from_dict(index_data)
        index.to_csv(output_path/self._index_name)

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
        if not in_place:
            data_dir = setup_data_dir(data_dir, overwrite)
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
            features = get_item_from_data(feature_dict, idx)
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

    def extend(
        self,
        feature_dict,
        in_file=True,
        in_place=True,
        data_dir=None,
        overwrite=False,
    ):
        if not in_place:
            data_dir = setup_data_dir(data_dir, overwrite)
            self.copy_to_dir(data_dir)
        else:
            data_dir = self.data_path

        _X = self.data_frame.X.to_list()
        _y = self.data_frame.y.to_list()

        for path in feature_dict.data_frame.X.values:
            out_fname = os.path.basename(path)
            out_path = os.path.join(data_dir, out_fname)
            _X.append(out_path)

        _y.extend(feature_dict.data_frame.y.to_list())

        feature_dict.copy_to_dir(data_dir, start_idx=len(self))

        index_file = pd.DataFrame.from_dict(dict(X=_X, y=_y))
        index_file.to_csv(os.path.join(data_dir, "index.csv"))
        if not in_place:
            self.data_frame = index_file

    def set_labels(self, labels):
        assert(len(self) == len(labels))
        self.data_frame.y.update(labels)
        self.data_frame.to_csv(self._index_file)

    def list_features(self):
        first_sample = self._get_x(0, collate=False, unshape=True)
        return list(first_sample.keys())

    def get_feature_shape(self, feature_name, throw=True):
        first_sample = self._get_x(0, collate=False, unshape=True)
        feature_shape = None
        if feature_name in first_sample:
            feature_shape = first_sample[feature_name].shape

        if throw and feature_shape is None:
            raise ValueError(
                f"Feature `{feature_name}` not found!"
            )
        return feature_shape

    def prepare_for_train(self, feature_set):
        self.ret_x = True
        self.ret_y = True
        self.cast = np.float32
        self.features = feature_set
        self.add_op(lambda x: x.squeeze(0), name="last", after_collate=True)

    def prepare_for_eval(self, feature_set):
        self.prepare_for_train(feature_set)
        self.ret_y = False


class EmbeddingsDataset(TorchDataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]
        return (self.X[index], self.y[index])
