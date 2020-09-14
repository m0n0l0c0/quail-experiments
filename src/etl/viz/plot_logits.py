"""Main module."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mcqa_utils import (
    Dataset,
    QASystemForMCOffline,
)

from mcqa_utils.utils import label_to_id
from mcqa_utils.mcqa_utils import get_masks_and_prefix


colors = ['b', 'g', 'r']


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
        ans.logits[label_to_id(ans.pred_label)] for ans in correct
    ]
    incorrect_logits = [
        ans.logits[label_to_id(ans.pred_label)] for ans in incorrect
    ]
    return correct_logits, incorrect_logits


def get_divided_logits(dataset, gold_answers, answers, mask):
    gold_reduced = dataset.reduce_by_mask(gold_answers, mask)
    answ_reduced = dataset.reduce_by_mask(answers, mask)
    corr_logits, incorr_logits = get_chosen_logits(
        gold_reduced, answ_reduced
    )
    return corr_logits, incorr_logits


def plot_logits_from_answers(dataset, gold_answers, answers, mask, color):
    corr_logits, incorr_logits = get_divided_logits(
        dataset, gold_answers, answers, mask
    )
    plt.plot(corr_logits, f'{color}o')
    plt.plot(incorr_logits, f'{color}x')


def main(data_path, preds_path):
    split = 'dev'
    dataset = Dataset(data_path=data_path, task='generic')
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    no_answer_text = 'not enough information'
    gold_answers = dataset.get_gold_answers(split, with_text_values=True)
    answers, missing = qa_system.get_answers(
        gold_answers,
        with_text_values=True,
        no_answer_text=no_answer_text,
    )
    masks, prefix = get_masks_and_prefix(
        dataset, no_answer_text, split
    )
    assert(len(missing) == 0)

    for pref, mask, color in zip(prefix, masks, colors[:len(masks)]):
        plot_logits_from_answers(dataset, gold_answers, answers, mask, color)
        plt.show()



if __name__ == '__main__':
    args = parse_flags()
    main(args.data_path, args.preds_path)
