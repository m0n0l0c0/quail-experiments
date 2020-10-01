"""Main module."""
import json
import argparse

from pathlib import Path
from mcqa_utils import QASystemForMCOffline, Dataset
from mcqa_utils.utils import label_to_id, id_to_label


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path', required=True, type=str,
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '-s', '--split', required=True, type=str,
        choices=['train', 'dev', 'test'],
        help='The split of the dataset to modify'
    )
    parser.add_argument(
        '-p', '--preds_path', required=True, type=str,
        help='Path to n_best_predictions file'
    )
    parser.add_argument(
        '-o', '--output_path', required=False, type=str,
        help='Output file to store the normalized logits'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Whether to overwrite the input file with the normalized logits'
    )
    parser.add_argument(
        '--no_answer_text', type=str, required=True,
        help='Text of an unaswerable question answer'
    )
    return parser.parse_args()


def setup_output_path(preds_path, output_path, overwrite):
    error_msg = (
        'Invalid output path!\n'
        'Pass `--overwrite` to save normalized logits to input file'
    )
    if output_path is None:
        if overwrite:
            output_path = preds_path
        else:
            raise ValueError(error_msg)
    else:
        output_path_obj = Path(output_path)
        if output_path_obj.exists() and not overwrite:
            raise ValueError(error_msg)
        else:
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def get_index_matching_text(sample, answer_text):
    match_text = answer_text.lower()
    matching_idx = 0
    while (
        matching_idx < len(sample.endings) and
        sample.endings[matching_idx].lower().find(match_text) == -1
    ):
        matching_idx += 1
    return matching_idx


def augment_probs(gold_answers, answers, no_answer_text):
    for gold, ans in zip(gold_answers, answers):
        matching_idx = get_index_matching_text(gold, no_answer_text)
        ans_idx = label_to_id(ans.pred_label)
        if matching_idx > ans_idx:
            ans.probs += [0.0]
        else:
            ans.probs.append(ans.probs[ans_idx])
            ans.probs[matching_idx] = 0.0
            ans.pred_label = id_to_label(label_to_id(ans.pred_label) + 1)
            if ans.label is not None:
                ans.label = id_to_label(label_to_id(ans.label) + 1)

    return answers


def main(data_path, preds_path, output_path, split, no_answer_text, overwrite):
    output_path = setup_output_path(preds_path, output_path, overwrite)
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    dataset = Dataset(data_path=data_path, task='generic')
    gold_answers = dataset.get_gold_answers(split, with_text_values=True)
    answers, _ = qa_system.get_answers(
        gold_answers,
        with_text_values=True,
        no_answer_text=no_answer_text
    )
    norm_answers = augment_probs(gold_answers, answers, no_answer_text)
    output_predictions = qa_system.unparse_predictions(norm_answers)

    with open(output_path, 'w') as fout:
        fout.write(json.dumps(output_predictions) + '\n')


if __name__ == '__main__':
    args = vars(parse_flags())
    main(**args)
