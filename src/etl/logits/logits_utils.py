import json
import numpy as np

from mcqa_utils import (
    Answer,
)


def exchange_logits_for_probs(path):
    data = json.load(open(path))
    for datapoint, values in data.items():
        for prob_dict in values:
            prob_dict['probs'] = prob_dict['logits'].copy()
            del prob_dict['logits']

    return data


def clone(answers):
    out_answers = []
    for ans in answers:
        out_answers.append(Answer.clone(ans))
    return out_answers


def normalize(answers, max_prob, field=None, exp=True):
    out_answers = clone(answers)
    if field is None:
        field = 'probs'
    if max_prob is None:
        max_prob = max([max(get_field(a, field)) for a in out_answers])
    if exp:
        max_prob = np.exp(max_prob)
    for ans in out_answers:
        probs = np.array(get_field(ans, field))
        if exp:
            probs = np.exp(get_field(ans, field))
        norm = probs/max_prob
        ans.__setattr__(field, norm.tolist())
    return out_answers


def get_field(answer, field):
    return answer.__getattribute__(field)
