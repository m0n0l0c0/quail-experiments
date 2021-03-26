import os
import sys
import json
import random
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from mcqa_utils import Dataset as McqaDataset
from mcqa_utils.utils import label_to_id, id_to_label
from mc_transformers.mc_transformers import softmax

base_path = os.path.dirname(os.path.dirname(__file__))
src_root = os.path.dirname(os.path.dirname(base_path))

sys.path.append(os.path.join(src_root, "processing"))
sys.path.append(os.path.join(src_root, "etl"))
sys.path.append(os.path.join(base_path, "classify"))
sys.path.append(os.path.join(base_path, "extract"))

from utils import load_classifier  # noqa: E402
from hyperp_utils import load_params  # noqa: E402
from dataset_class import Dataset  # noqa: E402
from classification import get_features_from_object  # noqa: E402
from classification import get_data_path_from_features  # noqa: E402
from choices.reformat_predictions import get_index_matching_text  # noqa: E402


class Args():
    def __init__(self, features, data_path=None):
        self.features = features
        self.data_path = data_path


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--classifier_path", required=True, default=None, type=str,
        help="Path to the classifier"
    )
    parser.add_argument(
        "-e", "--embeddings_path", required=True, default=None, type=str,
        help="Path to embeddings dataset directory (not the dataset itself)"
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


def apply_strategy(gold_answer, strategy_dict):
    if strategy_dict["type"] == "random":
        answer = random.choice(range(len(gold_answer.endings)))
    elif strategy_dict["type"] == "longest":
        answer = np.argmax(map(len, gold_answer.endings))
    elif strategy_dict["type"] == "no_answer":
        answer = get_index_matching_text(gold_answer, strategy_dict["extras"])

    return answer


def correct_sample(dataset, sample, match_idx):
    # remove endings matching no-answer-text
    for feat in dataset.list_features():
        shape = dataset.get_feature_shape(feat)
        if len(shape) and shape[0] == 4:
            sample[feat] = np.delete(sample[feat], match_idx)
    return sample


def get_classifier_from_model_answers(
    classifier,
    dataset,
    strategy_dict,
    gold_answers,
):
    mdl_answers = []
    cls_answers = []
    iterator = zip(gold_answers, dataset.iter(return_dict=True, y=False))
    tqdm_args = dict(desc="Applying classifier", total=len(dataset))
    for gold, x in tqdm(iterator, **tqdm_args):
        # classifiers are always trained on a 3-choice version of the dataset,
        # removing the no-answer-ending
        mdl_pred = int(np.argmax(softmax(x["logits"], axis=1)))
        mdl_answers.append(label_to_id(mdl_pred))
        match_idx = get_index_matching_text(gold, strategy_dict["extras"])
        x = correct_sample(dataset, x, match_idx)
        plain_x = dataset.destructure_sample(x)
        cls_answers.append(classifier.predict(plain_x))

    assert(len(mdl_answers) == len(cls_answers))
    return mdl_answers, cls_answers


def correct_model_with_classifier(
    mdl_answers,
    cls_answers,
    strategy_dict,
    gold_answers
):
    predictions = {}
    for gold, mdl_ans, cl_pred in zip(gold_answers, mdl_answers, cls_answers):
        # classifier says model is right
        if cl_pred == 1:
            pred_label = id_to_label(mdl_ans)
        else:
            pred_label = id_to_label(apply_strategy(gold, strategy_dict))

        predictions[gold.example_id] = pred_label

    return predictions


def get_path_from_features(classifier_path, data_path):
    params_path = Path(classifier_path).joinpath("params.yaml")
    params = load_params(params_path)
    features = get_features_from_object(params, allow_all_feats=True)
    embeddings_path = get_data_path_from_features(
        args=Args(features=features, data_path=data_path)
    )
    # no oversampling in dev set
    if "oversample_" in embeddings_path:
        embeddings_path = embeddings_path.replace("oversample_", "")

    return (
        embeddings_path,
        get_features_from_object(params, allow_all_feats=False)
    )


def main(
    classifier_path,
    embeddings_path,
    output_path,
    strategy,
    no_answer_text,
    data_path,
    split,
):
    print(f"Load gold answers from {data_path}")
    mcqa_dataset = McqaDataset(data_path=data_path, task='generic')
    gold_answers = mcqa_dataset.get_gold_answers(split, with_text_values=True)
    print(f"Load classifier from {classifier_path}")
    classifier = load_classifier(classifier_path)
    embeddings_path, features = get_path_from_features(
        classifier_path, embeddings_path
    )
    print(f"Load embeddings from {embeddings_path}")
    dataset = Dataset(
        data_path=embeddings_path, features=features
    )
    strategy_dict = dict(type=strategy, extras=no_answer_text)
    mdl_answers, cls_answers = get_classifier_from_model_answers(
        classifier, dataset, strategy_dict, gold_answers
    )
    full_predictions = correct_model_with_classifier(
        mdl_answers, cls_answers, strategy_dict, gold_answers
    )
    save_predictions(output_path, full_predictions)


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
