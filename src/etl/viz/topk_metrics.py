import os
import sys
import json
import numpy as np


data = json.load(open(sys.argv[1], "r"))
topk = int(sys.argv[2])

data_dir = os.path.dirname(sys.argv[1])
topk_path = os.path.join(data_dir, "topk.json")
zero_cl_path = os.path.join(data_dir, "class_0.json")
one_cl_path = os.path.join(data_dir, "class_1.json")

topk_field = "macro avg.f1-score"
best_cl_zero_fields = "0.f1-score"
best_cl_one_fields = "1.f1-score"


def field_access(data, key):
    aux = data
    for k in key.split("."):
        aux = aux[k]
    return aux


topk_mets = {}
keys = list(data.keys())

zero_vals = [field_access(elem, best_cl_zero_fields) for elem in data.values()]
zero_max = np.argmax(zero_vals)
best_cl_zero = {keys[zero_max]: data[keys[zero_max]]}

one_vals = [field_access(elem, best_cl_one_fields) for elem in data.values()]
one_max = np.argmax(one_vals)
best_cl_one = {keys[one_max]: data[keys[one_max]]}

vals = [field_access(elem, topk_field) for elem in data.values()]
topk_indices = np.argsort(vals, axis=None)[::-1][:topk].tolist()
for index in topk_indices:
    index_key = keys[index]
    topk_mets[index_key] = data[index_key]


with open(zero_cl_path, "w") as fout:
    fout.write(json.dumps(best_cl_zero) + "\n")

with open(one_cl_path, "w") as fout:
    fout.write(json.dumps(best_cl_one) + "\n")

with open(topk_path, "w") as fout:
    fout.write(json.dumps(topk_mets) + "\n")
