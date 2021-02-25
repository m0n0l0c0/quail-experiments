import os
import sys
import yaml
import collections.abc

from yaml import SafeLoader
from pathlib import Path
from collections import OrderedDict

sys.path.append("src/etl/embeddings/classify/")
from classification import get_data_path_from_features  # noqa: E402


def ordered_load(stream, Loader=SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def load_params(params_file):
    path = Path(os.getcwd()).absolute().joinpath(params_file)
    return ordered_load(open(path, "r"), SafeLoader)


def write_params(params, params_file):
    with open(params_file, "w") as fd:
        ordered_dump(params, stream=fd, Dumper=yaml.SafeDumper, indent=4)


def update_params(new_params, params):
    for k, v in new_params.items():
        if isinstance(v, collections.abc.Mapping):
            params[k] = update_params(v, params.get(k, {}))
        else:
            params[k] = v
    return params


def combination_to_params(comb):
    if comb is not None:
        comb = {
            "classification": {"pipeline": comb["pipeline"]},
            "features": {
                k: comb[k] for k in [
                    key for key in comb.keys() if key not in ["pipeline"]
                ]
            }
        }

    return comb


def validate_combination(args, combination):
    # dont allow combinations of all falsy values
    keys = [k for k in combination.keys() if k not in ["pipeline"]]
    valid = not all([not combination[k] for k in keys])
    if valid:
        data_path = get_data_path_from_features(args)
        valid = Path(data_path).joinpath("index.csv").exists()

    return valid
