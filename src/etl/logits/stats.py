"""Main module."""
import argparse
import numpy as np

from logits_utils import normalize, get_field
from mcqa_utils import (
    Dataset,
    QASystemForMCOffline,
)

from mcqa_utils.utils import label_to_id
from mcqa_utils.mcqa_utils import get_masks_and_prefix


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path', required=True, type=str,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '-p', '--preds_path', required=True, type=str,
        help='Path to n_best_predictions file'
    )
    parser.add_argument(
        '-s', '--split', default='dev', choices=['train', 'dev', 'test'],
        required=False, help='Split to evaluate from the dataset'
    )
    parser.add_argument(
        '--no_answer_text', type=str, required=True,
        help='Text of an unaswerable question answer'
    )
    parser.add_argument(
        '-pf', '--probs_field', type=str, required=False, default='logits',
        help='Field to use as `probs` field in prediction answers '
        '(default logits, but can be anything parsed in the answer)'
    )

    return parser.parse_args()


def get_correct_answers(gold_answers, answers):
    correct = []
    incorrect = []
    for gold, ans in zip(gold_answers, answers):
        # disable threshold, probs  mechanism
        if gold.get_answer() == label_to_id(ans.pred_label):
            correct.append(ans)
        else:
            incorrect.append(ans)
    return correct, incorrect


def stats_over_set(gold_answers, answers, chooser=None, field='logits'):
    if chooser is None:
        chooser = max_chooser

    probs = [chooser(get_field(a, field)) for a in answers]
    correct, incorrect = get_correct_answers(gold_answers, answers)
    corr_probs = [chooser(get_field(a, field)) for a in correct]
    incorr_probs = [chooser(get_field(a, field)) for a in incorrect]
    return [[
        np.mean(probs),
        np.mean(corr_probs),
        np.mean(incorr_probs),
    ], [
        np.std(probs),
        np.std(corr_probs),
        np.std(incorr_probs),
    ]]


def print_stats(means, stds, prefixes):
    for row in range(len(means[0])):
        stat = []
        for col in range(len(means)):
            stat.append(means[col][row])
            stat.append(stds[col][row])
        str_stats = '\t'.join(['{:.5f}'.format(a) for a in stat])
        print('{:25}'.format(prefixes[row]) + f'\t{str_stats}')


def do_stats(
    all_golds, all_sets, prefixes, max_value, chooser, field='logits'
):
    for gold_set, ans_set, ans_prefix in zip(all_golds, all_sets, prefixes):
        stat_means = []
        stat_stds = []
        means, stds = stats_over_set(
            gold_set, ans_set, chooser=chooser, field=field
        )
        stat_means.append(means)
        stat_stds.append(stds)

        norm_ans = normalize(ans_set, max_value, field=field, exp=True)
        means, stds = stats_over_set(
            gold_set, norm_ans, chooser=chooser, field=field
        )
        stat_means.append(means)
        stat_stds.append(stds)

        # means, stds = stats_over_set(
        #     gold_set,
        #     normalize(ans_set, max_value, exp=False),
        #     chooser=chooser
        # )
        # stat_means.append(means)
        # stat_stds.append(stds)

        prefs = [
            ans_prefix,
            f'Correct-{ans_prefix}',
            f'Incorrect-{ans_prefix}'
        ]
        print_stats(stat_means, stat_stds, prefs)
        print()


def second_chooser(array):
    sorted_arr = sorted(array, reverse=True)
    return sorted_arr[1]


def max_chooser(array):
    return max(array)


def main(data_path, preds_path, split, no_answer_text, probs_field):
    dataset = Dataset(data_path=data_path, task='generic')
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    gold_answers = dataset.get_gold_answers(split, with_text_values=True)
    answers, missing = qa_system.get_answers(
        gold_answers,
        with_text_values=True,
        no_answer_text=no_answer_text,
    )
    masks, prefix = get_masks_and_prefix(
        dataset, gold_answers, no_answer_text
    )
    assert(len(missing) == 0)

    gold_answerable = dataset.reduce_by_mask(gold_answers, masks[0])
    gold_unanswerable = dataset.reduce_by_mask(gold_answers, masks[1])
    answers_answerable = dataset.reduce_by_mask(answers, masks[0])
    answers_unanswerable = dataset.reduce_by_mask(answers, masks[1])

    all_golds = [gold_answers, gold_answerable, gold_unanswerable]
    all_sets = [answers, answers_answerable, answers_unanswerable]
    max_value = max([max(get_field(a, probs_field)) for a in answers])
    prefixes = ('global',) + prefix
    # print header
    header = "avg\tstd\tnrm.avg\tnrm.std"
    print('{:25}'.format("") + f'\t{header}')
    do_stats(
        all_golds, all_sets, prefixes, max_value,
        chooser=max_chooser, field=probs_field
    )
    prefixes = [f'Second-{pref}' for pref in prefixes]
    do_stats(
        all_golds, all_sets, prefixes, max_value,
        chooser=second_chooser, field=probs_field
    )


if __name__ == '__main__':
    args = parse_flags()
    main(**vars(args))
