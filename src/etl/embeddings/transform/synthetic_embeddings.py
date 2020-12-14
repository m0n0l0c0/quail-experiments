import os
import json
import random
import argparse

from tqdm import tqdm
from pathlib import Path
from mcqa_utils.dataset import Dataset
from mc_transformers.data_classes import InputExample

LOG = False


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", required=True, type=str,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "-n", "--num_samples", required=True, type=int,
        help="The number of samples to generate"
    )
    parser.add_argument(
        "-s", "--split", required=False, default="train", type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to synthetize data from"
    )
    parser.add_argument(
        "-t", "--task", default=None, required=False,
        help="Task to evaluate (default = generic). This "
        "is needed for the dataset processor (see geblanco/mc-transformers)"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store synthetic data"
    )
    args = parser.parse_args()
    if args.task is None:
        args.task = "generic"

    return args


def pick_random_examples(examples, num_samples):
    picked = []
    total_examples = len(examples)
    if num_samples > total_examples:
        raise ValueError(
            "Requested more examples than available"
        )
    indexes = random.choices(range(total_examples), k=num_samples + 1)
    progress = tqdm(indexes, disable=not LOG, desc="Choose samples")
    for idx in progress:
        picked.append(examples[idx])

    return picked


def synthesize_data(processor, examples):
    data = []
    # swap endings between examples
    all_endings = [end for ex in examples for end in ex.endings]
    if LOG:
        print("Shuffling...")

    random.shuffle(all_endings)
    cursor = 0
    progress = tqdm(examples, disable=not LOG, desc="Generate samples")
    for ex_idx, ex in enumerate(progress):
        # do not add labels
        new_example = InputExample(
            example_id=processor._encode_id("synth", ex_idx),
            question=ex.question,
            contexts=ex.contexts,
            endings=all_endings[cursor:cursor + len(ex.contexts)],
            label=ex.label
        )
        data.append(new_example)
        cursor += len(ex.contexts)

    return data


def save_dataset(dataset, data, output_dir, split, extend=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f"{split}.json"
    file_path = os.path.join(output_dir, file_name)
    if extend:
        examples = dataset.get_split(split)
        examples.extend(data)
    else:
        examples = data
    str_data = json.dumps(dataset.to_json(examples))
    if LOG:
        print(f"Saving data to '{file_path}'")
    with open(file_path, "w") as fout:
        fout.write(str_data + "\n")


def generate_synthetic_data(
    data_dir, output_dir, num_samples, split, task="generic", log=False
):
    global LOG
    LOG = log
    if LOG:
        print(f"Load data from '{data_dir}' (task = '{task}')")
    dataset = Dataset(data_path=data_dir, task=task)
    examples = dataset.get_split(split)
    chosen = pick_random_examples(examples, num_samples)
    data = synthesize_data(dataset.processor, chosen)
    save_dataset(dataset, data, output_dir, split)
    synthetic_out = os.path.join(output_dir, "synthetic")
    save_dataset(dataset, data, synthetic_out, split, extend=False)
    return synthetic_out


if __name__ == '__main__':
    LOG = True
    args = parse_flags()
    generate_synthetic_data(**vars(args))
