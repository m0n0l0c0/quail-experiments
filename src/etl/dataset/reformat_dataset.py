import os
import json
import argparse

from pathlib import Path
from mcqa_utils import Dataset, get_mask_matching_text
from mcqa_utils.utils import label_to_id, update_example


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path', required=True, type=str,
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '-o', '--output_dir', required=True, type=str,
        help='Directory to write the modified dataset split'
    )
    parser.add_argument(
        '-s', '--split', required=True, type=str,
        choices=['train', 'dev', 'test'],
        help='The split of the dataset to modify'
    )
    parser.add_argument(
        '--no_answer_text', type=str, required=True,
        help='Text of an unaswerable question answer'
    )
    return parser.parse_args()


def remove_extra_option(examples, answer_text):
    match_text = answer_text.lower()
    ret_examples = []
    for sample in examples:
        matching_idx = 0
        while (
            matching_idx < len(sample.endings) and
            sample.endings[matching_idx].lower().find(match_text) == -1
        ):
            matching_idx += 1

        ans_index = label_to_id(sample.label)
        if matching_idx < len(sample.endings):
            if ans_index == matching_idx:
                print(sample)
                raise ValueError(
                    'Something went really wrong, the answer to delete '
                    'is the correct one, did you masked the wrong way?'
                )
            del sample.endings[matching_idx]

        if ans_index > matching_idx:
            ans_index -= 1
            ex = update_example(sample, label=ans_index)
            ret_examples.append(ex)
        else:
            ret_examples.append(sample)

    return ret_examples


def dump_examples(dataset, examples, output_dir, split):
    json_data = dataset.to_json(examples)
    json_str = json.dumps(json_data, ensure_ascii=False) + '\n'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, f'{split}.json'), 'w') as fout:
        fout.write(json_str)


def main(data_path, output_dir, split, no_answer_text):
    dataset = Dataset(data_path=data_path, task='generic')
    answerable_mask = get_mask_matching_text(no_answer_text, match=False)
    examples = dataset.get_split(split)
    answerable_examples = dataset.reduce_by_mask(
        examples, answerable_mask
    )
    stats_str = f'({len(answerable_examples)}/{len(examples)}: '
    stats_str += '{:.2f}%)'.format(
        (len(answerable_examples)/len(examples)) * 100
    )
    print(f'{stats_str} of examples are valid')
    mod_examples = remove_extra_option(answerable_examples, no_answer_text)
    dump_examples(dataset, mod_examples, output_dir, split)


if __name__ == '__main__':
    args = parse_flags()
    main(**vars(args))
