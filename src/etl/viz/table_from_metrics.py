import re
import json
import pandas as pd


def fix_key_name(key):
    # key comes as: data/metrics/bert/generic/race-multibert-train/...
    # get: race-multibert-train
    key_splt = key.split('/')
    return key_splt[4]


def print_df(df, print_keys):
    print_str = [
        re.sub(space_reg, '\t', line)
        for line in
        df.loc[print_keys[0]:print_keys[1]].T.to_string().split('\n')
    ]
    print('\n'.join(print_str) + '\n')


file = './metrics.json'
data = json.load(open(file, 'r'))
data = list(data.values())[0]
data_keys = list(data.keys())

for key in data_keys:
    data[fix_key_name(key)] = data[key]
    del data[key]

print_keys = ['C_at_1', 'avg_incorrect']
separate_keys = ['has_ans', 'no_has_ans', 'C_at_1_threshold', 'avg_threshold']
sub_keys = ['has_ans', 'no_has_ans']
space_reg = re.compile(r'\s+')

metrics_df = pd.DataFrame(data)
print_df(metrics_df, print_keys)

separate_dicts = {}
model_names = list(data.keys())

for sep_key in separate_keys:
    separate_dicts[sep_key] = pd.DataFrame({
        model_name: data[model_name][sep_key]
        for model_name in model_names
    })
    print(sep_key)
    print_df(separate_dicts[sep_key], print_keys)

    for sub in sub_keys:
        if sub in separate_dicts[sep_key].index:
            sep_sub_key = f'{sep_key}_{sub}'
            # nested dict
            separate_dicts[sep_sub_key] = pd.DataFrame.from_dict(
                separate_dicts[sep_key].loc[sub].to_dict()
            )
            print(sep_sub_key)
            print_df(separate_dicts[sep_sub_key], print_keys)
