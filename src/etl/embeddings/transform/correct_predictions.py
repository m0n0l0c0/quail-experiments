import os
import sys
import json
import random
import argparse
import numpy as np

from pathlib import Path

from transformers import is_tf_available

from mcqa_utils import Dataset as McqaDataset
from mcqa_utils.utils import label_to_id, id_to_label
from mc_transformers.mc_transformers import softmax

base_path = os.path.dirname(os.path.dirname(__file__))
src_root = os.path.dirname(os.path.dirname(base_path))

sys.path.append(os.path.join(src_root, "etl"))
sys.path.append(os.path.join(base_path, "classify"))
sys.path.append(os.path.join(base_path, "extract"))

from utils import load_classifier  # noqa: E402
from dataset_class import Dataset  # noqa: E402
from choices.reformat_predictions import get_index_matching_text  # noqa: E402


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--classifier_path", required=True, default=None, type=str,
        help="Path to the trained classifier"
    )
    parser.add_argument(
        "-e", "--embeddings_path", required=True, default=None, type=str,
        help="Path to embeddings dataset"
    )
    parser.add_argument(
        "-o", "--output_path", required=False, type=str,
        help="Path to store corrected predictions"
    )
    parser.add_argument(
        "--strategy", required=False, default="no_answer", type=str,
        choices=["no_answer", "random", "longest"],
        help="Strategy to apply over answers where the model is wrong "
        "(by the classifier criteria)"
    )
    parser.add_argument(
        "--no_answer_text", required=False, default=None, type=str,
        help="Text of an unaswerable question answer (only necessary "
        "for no_answer strategy)"
    )
    parser.add_argument(
        "-d", "--data_path", required=True, default=None, type=str,
        help="Path to original dataset to extract gold answers"
    )
    parser.add_argument(
        "-s", "--split", required=False, default="dev", type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to extract embeddings from"
    )

    args = parser.parse_args()
    if args.strategy == "no_answer" and args.no_answer_text is None:
        raise ValueError(
            "When `no_answer` strategy is selected, you must provide "
            "no_answer_text in order to find the correct answer!"
        )
    return args


def save_predictions(output_path, predictions):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fout:
        fout.write(json.dumps(predictions) + "\n")


def get_dataset_from_embeddings(embeddings, logits, labels):
    num_samples = logits.shape[0]
    num_choices = logits.shape[1]

    embeddings = embeddings.reshape(
        num_samples, num_choices, *embeddings.shape[1:]
    )

    return dict(
        embeddings=embeddings,
        logits=logits,
        labels=labels,
    )


def get_predictions_from_classifier(dataset, classifier_path, autogoal):
    X_test, y_test = get_x_y_from_dict(dataset)
    classifier = load_classifier(classifier_path)
    return classifier.predict(X_test)


def apply_strategy(gold_answer, strategy_dict):
    if strategy_dict["type"] == "random":
        answer = random.choice(range(len(gold_answer.endings)))
    elif strategy_dict["type"] == "longest":
        answer = np.argmax(map(len, gold_answer.endings))
    elif strategy_dict["type"] == "no_answer":
        answer = get_index_matching_text(gold_answer, strategy_dict["extras"])

    return answer


def correct_model_with_classifier(
    classifier,
    dataset,
    strategy_dict,
    gold_answers
):
    mdl_answers = []
    cls_answers = []
    data_iter = dataset.iter(return_dict=True, x=True, y=True)
    for gold, (x, y) in zip(gold_answers, data_iter):
        label = y["label"]
        assert(id_to_label(gold.label) == id_to_label(label))
        mdl_pred = int(np.argmax(softmax(x["logits"], axis=1)))
        mdl_answers.append(label_to_id(mdl_pred))
        cls_answers.append(classifier.predict(x["embeddings"]))

    assert(len(mdl_answers) == len(cls_answers))

    predictions = {}
    for gold, cl_pred, probs in zip(gold_answers, mdl_answers, cls_answers):
        # classifier says model is right
        if cl_pred == 1:
            pred_label = id_to_label(gold)
        else:
            pred_label = id_to_label(apply_strategy(gold, strategy_dict))

        predictions[gold.example_id] = pred_label

    return predictions


def main(
    classifier_path,
    embeddings_path,
    output_path,
    strategy,
    no_answer_text,
    data_path,
    split,
):
    mcqa_dataset = McqaDataset(data_path=data_path, task='generic')
    gold_answers = mcqa_dataset.get_gold_answers(split, with_text_values=True)
    print(f"Load gold answers from {data_path}")
    classifier = load_classifier(classifier_path)
    print(f"Load classifier from {classifier_path}")
    dataset = Dataset(data_path=embeddings_path)
    print(f"Load embeddings from {embeddings_path}")
    strategy_dict = dict(type=strategy, extras=no_answer_text)
    full_predictions = correct_model_with_classifier(
        classifier, dataset, strategy_dict, gold_answers
    )
    save_predictions(output_path, full_predictions)


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
