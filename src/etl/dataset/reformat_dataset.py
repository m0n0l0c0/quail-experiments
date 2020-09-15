from mcqa_utils import Dataset, get_mask_matching_text


def remove_extra_option(examples, answer_text):
    match_text = answer_text.lower()
    for sample in examples:
        matching_idx = -1
        for matching_idx in range(len(sample.endings)):
            end_text = sample.endings[matching_idx].lower()
            if end_text.find(match_text) != -1:
                break
        if matching_idx > 0 and matching_idx < len(sample.endings):
            del sample.endings[matching_idx]
    return examples


def dump_examples(examples, output_path):
    # ToDo:= make processor dump examples
    pass


def main(data_path, output_data_path, no_answer_text):
    split = 'train'
    dataset = Dataset(data_path=data_path, task='generic')
    answerable_mask = get_mask_matching_text(no_answer_text, match=False)
    train_examples = dataset.get_train_examples()
    answerable_examples = dataset.reduce_by_mask(
        train_examples, answerable_mask
    )
    print(f'{len(answerable_examples)/len(train_examples)}% of examples are '
        f'valid, {len(answerable_examples)} in total'
    )
    mod_examples = remove_extra_option(answerable_examples, no_answer_text)
    dump_examples(mod_examples, output_data_path)