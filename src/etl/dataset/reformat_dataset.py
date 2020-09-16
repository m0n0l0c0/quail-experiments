import json
import argparse
from mcqa_utils import Dataset, get_mask_matching_text
from mcqa_utils.utils import label_to_id, update_example


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path', required=True, type=str,
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '-o', '--output_path', required=True, type=str,
        help='File path to leave the modified dataset'
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
        matching_idx = -1
        for matching_idx in range(len(sample.endings)):
            end_text = sample.endings[matching_idx].lower()
            if end_text.find(match_text) != -1:
                break

        if matching_idx > 0 and matching_idx < len(sample.endings):
            del sample.endings[matching_idx]

        ans_index = label_to_id(sample.label)
        if ans_index == matching_idx:
            raise ValueError(
                'Something went really wrong, the answer to delete '
                'is the correct one, did you masked the wrong way?'
            )
        elif ans_index > matching_idx:
            ans_index -= 1
            ex = update_example(sample, label=ans_index)
            ret_examples.append(ex)
        else:
            ret_examples.append(sample)

    return ret_examples


def dump_examples(dataset, examples, output_path):
    json_data = dataset.to_json(examples)
    json_str = json.dumps(json_data, ensure_ascii=False) + '\n'
    with open(output_path, 'w') as fout:
        fout.write(json_str)


def main(data_path, output_path, split, no_answer_text):
    dataset = Dataset(data_path=data_path, task='generic')
    answerable_mask = get_mask_matching_text(no_answer_text, match=False)
    examples = dataset.get_split(split)
    answerable_examples = dataset.reduce_by_mask(
        examples, answerable_mask
    )
    print(f'{len(answerable_examples)/len(examples)}% of examples are '
        f'valid, {len(answerable_examples)} in total'
    )
    mod_examples = remove_extra_option(answerable_examples, no_answer_text)
    dump_examples(dataset, mod_examples, output_path)


if __name__ == '__main__':
    args = parse_flags()
    main(**vars(args))