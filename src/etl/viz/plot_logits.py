"""Main module."""
import os
import sys
import argparse
import matplotlib.pyplot as plt

from mcqa_utils import (
    Dataset,
    QASystemForMCOffline,
)

from mcqa_utils.utils import label_to_id
from mcqa_utils.mcqa_utils import get_masks_and_prefix

basedir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(basedir, 'logits'))

from logits_utils import get_field  # noqa: E402


colors = ['b', 'g', 'r']
draw_field = 'logits'


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
        '--no_answer_text', type=str, required=False, default=None,
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


def get_chosen_logits(gold_answers, answers):
    correct, incorrect = get_correct_answers(gold_answers, answers)
    correct_logits = [
        get_field(ans, draw_field)[label_to_id(ans.pred_label)]
        for ans in correct
    ]
    incorrect_logits = [
        get_field(ans, draw_field)[label_to_id(ans.pred_label)]
        for ans in incorrect
    ]
    return correct_logits, incorrect_logits


def get_divided_logits(dataset, gold_answers, answers, mask):
    if mask is None:
        return get_chosen_logits(gold_answers, answers)

    gold_reduced = dataset.reduce_by_mask(gold_answers, mask)
    answ_reduced = dataset.reduce_by_mask(answers, mask)
    corr_logits, incorr_logits = get_chosen_logits(
        gold_reduced, answ_reduced
    )
    return corr_logits, incorr_logits


def plot_logits_from_answers(dataset, gold_answers, answers, mask=None):
    corr_logits, incorr_logits = get_divided_logits(
        dataset, gold_answers, answers, mask
    )
    plt.plot(corr_logits, 'go')
    plt.plot(incorr_logits, 'rx')


def main(data_path, preds_path, split, no_answer_text, probs_field):
    global draw_field
    if probs_field is not None:
        draw_field = probs_field
    dataset = Dataset(data_path=data_path, task='generic')
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    if no_answer_text is not None:
        gold_answers = dataset.get_gold_answers(split, with_text_values=True)
        answers, missing = qa_system.get_answers(
            gold_answers,
            with_text_values=True,
            no_answer_text=no_answer_text,
        )
        masks, prefix = get_masks_and_prefix(
            dataset, gold_answers, no_answer_text
        )
    else:
        gold_answers = dataset.get_gold_answers(split)
        answers, missing = qa_system.get_answers(gold_answers)
        masks = None

    assert(len(missing) == 0)

    if masks is None:
        plot_logits_from_answers(dataset, gold_answers, answers)
        plt.show()
    else:
        for pref, mask in zip(prefix, masks):
            plot_logits_from_answers(dataset, gold_answers, answers, mask)
            plt.show()


if __name__ == '__main__':
    args = parse_flags()
    main(**vars(args))
