"""Main module."""
import json
import argparse
import numpy as np

from pathlib import Path
from mcqa_utils import QASystemForMCOffline


def parse_flags():
    parser = argparse.ArgumentParser()
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
        '-s', '--size', required=True, type=int,
        help='Total size for probs field (new dimensions will be 0)'
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


def augment_probs(answers, size):
    for ans in answers:
        ans.probs += [0.0] * (size - len(ans.probs))
    return answers


def main(preds_path, output_path, overwrite, size):
    output_path = setup_output_path(preds_path, output_path, overwrite)
    qa_system = QASystemForMCOffline(answers_path=preds_path)
    all_answers = qa_system.get_all_answers()
    norm_answers = augment_probs(all_answers, size)
    output_predictions = qa_system.unparse_predictions(norm_answers)

    with open(output_path, 'w') as fout:
        fout.write(json.dumps(output_predictions) + '\n')


if __name__ == '__main__':
    args = parse_flags()
    main(args.preds_path, args.output_path, args.overwrite, args.size)
