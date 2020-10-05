"""Main module."""
import json
import argparse

from pathlib import Path
from mcqa_utils import QASystemForMCOffline, Dataset, get_mask_matching_text
from mcqa_utils.utils import label_to_id, id_to_label


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path', required=False, default=None, type=str,
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '-s', '--split', required=False, default='dev', type=str,
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
    parser.add_argument(
        '--keep_matching_text', action='store_true',
        help='Whether to keep the examples with the answer matching the given'
        ' text (default is to remove those examples)'
    )
    parser.add_argument(
        '--index_list_path', type=str, default=None, required=False,
        help='Index list to restore predictions, used when not working with'
        ' gold standard'
    )
    args = parser.parse_args()
    if (
        (
            args.index_list_path is None and
            (args.data_path is None and args.split is None)
        ) or
        (args.index_list_path is not None and args.data_path is not None)
    ):
        raise ValueError(
            'You must specify either data_path with gold standard or '
            'index list to restore predictions (not both or none of them)'
        )
    return args


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


def augment_probs(
    gold_answers, answers, no_answer_text, index_list=None, fix_label=False
):
    by_index_list = False
    if index_list is not None:
        by_index_list = True
        gold_answers = index_list
    if fix_label and by_index_list:
        raise ValueError(
            'Cannot fix labels when working with index lists '
            '(pass keep_matching_text=False) to correct this'
        )
    for gold, ans in zip(gold_answers, answers):
        if by_index_list:
            matching_idx = label_to_id(gold)
        else:
            matching_idx = get_index_matching_text(gold, no_answer_text)
        ans_idx = label_to_id(ans.pred_label)
        if matching_idx > ans_idx:
            ans.probs.append(0.0)
        else:
            ans.probs.insert(matching_idx, 0.0)
            ans.pred_label = id_to_label(ans_idx + 1)
        if fix_label and ans.label is not None:
            ans.label = id_to_label(label_to_id(gold.label))

        # align ids
        if not by_index_list:
            ans.example_id = gold.example_id
    return answers


def main(
    data_path,
    preds_path,
    output_path,
    split,
    no_answer_text,
    overwrite,
    index_list_path,
    keep_matching_text,
):
    output_path = setup_output_path(preds_path, output_path, overwrite)
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    answers = qa_system.get_all_answers()
    augment_args = dict(
        answers=answers,
        no_answer_text=no_answer_text,
        fix_label=(not keep_matching_text)
    )
    if data_path is not None:
        dataset = Dataset(data_path=data_path, task='generic')
        gold_answers = dataset.get_gold_answers(split, with_text_values=True)
        if not keep_matching_text:
            answerable_mask = get_mask_matching_text(
                no_answer_text, match=False
            )
            gold_reduced = dataset.reduce_by_mask(
                gold_answers, answerable_mask
            )
            assert(len(gold_reduced) == len(answers))
            augment_args.update(gold_answers=gold_reduced)
        else:
            augment_args.update(gold_answers=gold_answers)

        norm_answers = augment_probs(**augment_args)
        output_predictions = qa_system.unparse_predictions_with_alignment(
            gold_answers, norm_answers
        )
    else:
        index_list = json.load(open(index_list_path, 'r')).values()
        augment_args.update(gold_answers=None, index_list=index_list)
        norm_answers = augment_probs(**augment_args)
        output_predictions = qa_system.unparse_predictions(norm_answers)

    with open(output_path, 'w') as fout:
        fout.write(json.dumps(output_predictions) + '\n')


if __name__ == '__main__':
    args = vars(parse_flags())
    main(**args)
