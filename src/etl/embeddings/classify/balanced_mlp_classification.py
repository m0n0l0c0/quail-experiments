import pickle
import argparse
import numpy as np

from pathlib import Path
from collections import Counter

from sklearn.metrics import classification_report

from autogoal.search import PESearch
from autogoal.ml.metrics import accuracy
from torch.utils.data.dataloader import DataLoader

from mlp_classifier import MLPClassifier, EmbeddingsDataset, get_pipeline
from balanced_sampling import balanced_sampling_iter
from classification import (
    get_dataset,
    get_splits,
    get_x_y_from_dict,
    get_loggers,
    normalize_dataset,
)


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Path to the dataset"
    )
    parser.add_argument(
        "-bs", "--batch_size", required=False, type=int, default=1024,
        help="The batch size for train/predict"
    )
    parser.add_argument(
        "--lr", required=False, type=float, default=0.01,
        help="Learning rate for optimization algorithm"
    )
    parser.add_argument(
        "--epochs", required=False, type=int, default=50,
        help="Training epochs for MLP classifier"
    )
    parser.add_argument(
        "-ts", "--test_size", required=False, type=float, default=0.33,
        help="The percentage of examples to use for test"
    )
    parser.add_argument(
        "-f", "--features", required=False, type=str, nargs="*", default=None,
        help="Features used to train the classifier"
    )
    parser.add_argument(
        "-a", "--autogoal", action="store_true",
        help="Whether to perform hyper search with autogoal (dafault False)"
    )
    parser.add_argument(
        "-i", "--iterations", required=False, type=int, default=100,
        help="Iterations to run for autogoal training"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store the models"
    )
    return parser.parse_args()


def get_dataset_rounds(train_dict):
    y_train = train_dict["labels"]
    props = list(Counter(y_train).values())
    max_nof, min_nof = max(props), min(props)
    return round(max_nof / min_nof)


def get_hidden_size(train_dict, features=None):
    X_train, _ = get_x_y_from_dict(train_dict, features=features)
    return X_train.reshape(X_train.shape[0], -1).shape[-1]


def train_classifier(
    classifier,
    train_epochs,
    dataset_rounds,
    train_dict,
    test_dict,
    feature_set,
    batch_size,
    print_result=True,
):
    if not classifier.is_initialized:
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier._initialize(hidden_size)

    for epoch in range(1, train_epochs + 1):
        classifier.model.train()
        epoch_loss = 0
        epoch_acc = 0
        for X_train, y_train in balanced_sampling_iter(
            dataset_rounds,
            train_dict,
            feature_set,
            dont_reshape=False
        ):
            embed_dataset = EmbeddingsDataset(X_train, y_train)
            train_loader = DataLoader(
                dataset=embed_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            for batch in train_loader:
                X_train, y_train = batch
                loss, acc = classifier.fit(X_train, y_train)
                epoch_loss += (loss / dataset_rounds)
                epoch_acc += (acc / dataset_rounds)

        if print_result:
            eval_acc, y_test, y_preds = eval_classifier(
                classifier,
                test_dict,
                feature_set,
                batch_size,
                print_result=False,
                return_y=True
            )
            status = f"Epoch {epoch+0:03}: | "
            status += f"Loss: {epoch_loss/len(train_loader):.5f} | "
            status += f"Acc: {epoch_acc/len(train_loader):.3f} | "
            status += f"Eval Acc: {eval_acc:.3f} | "
            print(status)

            if epoch % 20 == 0:
                print(classification_report(y_test, y_preds))


def eval_classifier(
    classifier,
    test_dict,
    feature_set,
    batch_size,
    score_fn=accuracy,
    print_result=True,
    return_y=False,
):
    y_preds_list = None
    classifier.model.eval()
    X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)
    embed_dataset = EmbeddingsDataset(X_test)
    test_loader = DataLoader(
        dataset=embed_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    for batch in test_loader:
        y_preds = classifier.predict(batch)
        if y_preds_list is None:
            y_preds_list = y_preds
        else:
            y_preds_list = np.concatenate([y_preds_list, y_preds], axis=0)

    y_preds_list = y_preds_list.squeeze()
    if print_result:
        print(classification_report(y_test, y_preds_list))

    output = score_fn(y_test, y_preds_list)
    if return_y:
        output = (output, y_test, y_preds_list)

    return output


def make_fn(
    args,
    train_dict,
    test_dict,
    feature_set,
    score_fn,
):
    dataset_rounds = get_dataset_rounds(train_dict)
    train_data = dict(
        train_epochs=args.epochs,
        dataset_rounds=dataset_rounds,
        train_dict=train_dict,
        test_dict=test_dict,
        feature_set=feature_set,
        batch_size=args.batch_size,
        print_result=False,
    )

    eval_data = dict(
        test_dict=test_dict,
        feature_set=feature_set,
        batch_size=args.batch_size,
        score_fn=score_fn,
        print_result=False,
    )

    def fitness(pipeline):
        train_classifier(pipeline.classifier, **train_data)
        return eval_classifier(pipeline.classifier, **eval_data)

    return fitness


def setup_pipeline(args, train_dict, test_dict, feature_set, score_fn):
    pipeline = get_pipeline(log_grammar=True)
    fitness_fn = make_fn(
        args,
        train_dict,
        test_dict,
        feature_set,
        score_fn,
    )
    return PESearch(
        pipeline,
        fitness_fn,
        pop_size=5,
        selection=2,
        evaluation_timeout=1800,
        memory_limit=64 * (1024**3),
        early_stop=True,
    )


def autogoal_train(args, train_dict, test_dict, features, score_fn):
    loggers = get_loggers(args.output_dir)
    for i, feature_set in enumerate(features):
        pipeline = setup_pipeline(
            args,
            train_dict,
            test_dict,
            feature_set,
            score_fn,
        )
        best_pipe, score = pipeline.run(args.iterations, logger=loggers)
        print(f"Pipe {best_pipe}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        classifier_fname = f"{args.output_dir}/classifier_{i}.pkl"
        with open(classifier_fname, "wb") as fout:
            pickle.dump(best_pipe, fout)


def std_train(args, train_dict, test_dict, features, score_fn):
    for i, feature_set in enumerate(features):
        print(f"Training with features: {feature_set}")
        dataset_rounds = get_dataset_rounds(train_dict)
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier = MLPClassifier(lr=args.lr)
        classifier._initialize(hidden_size)
        train_data = dict(
            train_epochs=args.epochs,
            dataset_rounds=dataset_rounds,
            train_dict=train_dict,
            test_dict=test_dict,
            feature_set=feature_set,
            batch_size=args.batch_size
        )
        test_data = dict(
            test_dict=test_dict,
            feature_set=feature_set,
            batch_size=args.batch_size,
            score_fn=score_fn,
        )
        train_classifier(classifier, **train_data)
        eval_classifier(classifier, **test_data)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        classifier_fname = f"{args.output_dir}/classifier_feature_{i}.pkl"
        with open(classifier_fname, "wb") as fout:
            pickle.dump(classifier, fout)

        # ToDo := Save parameters
        print(f"Saved model to: {classifier_fname}")


def main(args):
    print(f"Loading data from {args.data_path}")

    dataset = get_dataset(args.data_path)
    features = [args.features] if args.features is not None else [
        ["embeddings"],
        ["embeddings", "logits"]
    ]

    dataset = normalize_dataset(dataset, features[-1])
    train_dict, test_dict = get_splits(dataset, test_size=args.test_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_fn = autogoal_train if args.autogoal else std_train
    train_fn(args, train_dict, test_dict, features, accuracy)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
