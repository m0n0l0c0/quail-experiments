import os
import sys
import json
import torch
import random
import argparse
import numpy as np

from pathlib import Path

from transformers import is_tf_available

from mcqa_utils import Dataset
from mcqa_utils.utils import id_to_label
from mc_transformers.mc_transformers import softmax
from src.etl.embeddings.classify.pipeline import load_pipeline
from src.etl.embeddings.classify.classification import get_x_y_from_dict
from src.etl.embeddings.extract_embeddings import embed_dataset, mc_setup

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)
from choices.reformat_predictions import get_index_matching_text  # noqa: E402

if is_tf_available():
    # Force no unnecessary allocation
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


os.environ.update(**{"WANDB_DISABLED": "true"})


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--classifier_path", required=True, default=None, type=str,
        help="Path to the trained classifier"
    )
    parser.add_argument(
        "-a", "--args_file", required=True, type=str,
        help="Arguments in json used to work with the model"
    )
    parser.add_argument(
        "-g", "--gpu", required=False, default=0, type=int,
        help="GPU to use (default to 0)"
    )
    parser.add_argument(
        "-s", "--split", required=False, default="dev", type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to extract embeddings from"
    )
    parser.add_argument(
        "-o", "--output_path", required=False, type=str,
        help="Path to store corrected predictions"
    )
    parser.add_argument(
        "--autogoal", action="store_true",
        help="Whether the classifier was trained with autogoal or custom "
        "pipeline"
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
    args = parser.parse_args()
    if args.strategy == "no_answer" and args.no_answer_text is None:
        raise ValueError(
            "When `no_answer` strategy is selected, you must provide "
            "no_answer_text in order to find the correct answer!"
        )
    return args


def save_data(output_path, predictions):
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
    sys.path.append(os.path.join(os.getcwd(), 'src/etl/embeddings/classify/'))
    classifier = load_pipeline(classifier_path, autogoal_pipe=autogoal)
    return classifier.predict(X_test)


def apply_strategy(gold_answer, strategy_dict):
    if strategy_dict["type"] == "random":
        answer = random.choice(range(len(gold_answer.endings)))
    elif strategy_dict["type"] == "longest":
        answer = np.argmax(map(len, gold_answer.endings))
    elif strategy_dict["type"] == "no_answer":
        answer = get_index_matching_text(gold_answer, strategy_dict["extras"])

    return answer


def correct_model_with_classifier(all_args, preds_dict, strategy_dict, split):
    class_preds = preds_dict['classifier']
    model_preds = softmax(preds_dict['model'], axis=1)
    data_args = all_args["data_args"]

    mcqa_dataset = Dataset(data_path=data_args.data_dir, task="generic")
    gold_answers = mcqa_dataset.get_gold_answers(split, with_text_values=True)

    assert(len(gold_answers) == len(class_preds) == len(model_preds))

    predictions = {}
    for gold, cl_pred, probs in zip(gold_answers, class_preds, model_preds):
        # classifier says model is right
        if cl_pred == 1:
            pred_label = id_to_label(np.argmax(probs))
        else:
            pred_label = id_to_label(apply_strategy(gold, strategy_dict))

        predictions[gold.example_id] = pred_label

    return predictions


def main(
    classifier_path,
    args_file,
    gpu,
    split,
    output_path,
    autogoal,
    strategy,
    no_answer_text,
):
    all_args, model, trainer, eval_dataset = mc_setup(args_file, split)
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    device = torch.device("cuda", index=gpu)
    embeddings, logits, labels = embed_dataset(model, eval_dataloader, device)
    dataset = get_dataset_from_embeddings(embeddings, logits, labels)
    predictions = get_predictions_from_classifier(
        dataset, classifier_path, autogoal
    )
    predictions_dict = dict(classifier=predictions, model=logits)
    strategy_dict = dict(type=strategy, extras=no_answer_text)
    full_predictions = correct_model_with_classifier(
        all_args, predictions_dict, strategy_dict, split
    )
    save_data(output_path, full_predictions)


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
