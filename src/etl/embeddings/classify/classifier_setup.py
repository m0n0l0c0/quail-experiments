import torch
import random
import numpy as np

from pipeline import get_pipeline
from dataset_class import Dataset
from balanced_sampling import balanced_resample
from dataset import get_x_y_from_dict
from classifiers import MLPClassifier
from classifier_mlp_train_eval import (
    get_hidden_size,
    train_classifier as mlp_train,
    eval_classifier as mlp_eval,
)

from autogoal.utils import Gb
from autogoal.search import PESearch
from autogoal.ml import AutoML
from autogoal.kb import (
    MatrixContinuousDense,
    CategoricalVector,
)

GPU_DEVICE = None


def setup_gpu_device(gpu_num):
    global GPU_DEVICE
    GPU_DEVICE = torch.device("cuda", index=gpu_num)


def make_balanced_fn(args, train_dict, test_dict, feature_set, score_fn):
    if isinstance(train_dict, Dataset):
        raise ValueError(
            "Cannot do balanced sampling on a Dataset instance, load raw data"
            " or balance outside classification process"
        )
    X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)
    X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
    X_train, y_train = balanced_resample(
        seed=random.choice(list(range(1000))),
        X_train=X_train,
        y_train=y_train,
    )

    def fitness(pipeline):
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        return score_fn(y_pred, y_test)

    return fitness


def make_fn(args, train_dict, test_dict, feature_set, score_fn):
    if isinstance(train_dict, Dataset):
        X_train, y_train = train_dict.get_x_y_from_dict(features=feature_set)
        X_test, y_test = test_dict.get_x_y_from_dict(features=feature_set)
    else:
        X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
        X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)

    def fitness(pipeline):
        X_train_list = X_train
        y_train_list = y_train
        X_test_list = X_test
        y_test_list = y_test

        # best effor to make it work with AutoGoal contrib modules
        if args.autogoal:
            X_train_list = np.array(X_train.to_list())
            y_train_list = np.array(y_train.to_list())

        pipeline.fit(X_train_list, y_train_list)

        if isinstance(X_test, Dataset):
            X_test_list = np.array(X_test.to_list())
            y_test_list = np.array(y_test.to_list())

        y_pred = pipeline.predict(X_test_list)
        return score_fn(y_test_list, y_pred)

    return fitness


def make_mlp_fn(args, train_dict, test_dict, feature_set, score_fn):
    train_data = dict(
        train_epochs=args.epochs,
        train_dict=train_dict,
        test_dict=test_dict,
        feature_set=feature_set,
        batch_size=args.batch_size,
        score_fn=score_fn,
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
        if args.train:
            mlp_train(pipeline, **train_data)

        y_test, y_pred = mlp_eval(pipeline, **eval_data)
        return score_fn(y_test, y_pred)

    return fitness


def setup_pe_pipeline(args, train_dict, test_dict, feature_set, score_fn):
    pipeline = get_pipeline(pipe_type=args.pipeline, log_grammar=True)
    if args.pipeline == "mlp":
        maker = make_mlp_fn
    elif args.balanced:
        maker = make_balanced_fn
    else:
        maker = make_fn

    fitness_fn = maker(args, train_dict, test_dict, feature_set, score_fn)

    classifier = PESearch(
        pipeline,
        fitness_fn,
        pop_size=args.popsize,
        selection=args.selection,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * Gb,
        early_stop=args.early_stop,
        random_state=args.seed,
    )
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def setup_automl(args, train_dict, test_dict, feature_set, score_fn):
    classifier = AutoML(
        input=MatrixContinuousDense(),
        output=CategoricalVector(),
        search_algorithm=PESearch,
        search_iterations=args.iterations,
        score_metric=score_fn,
        random_state=args.seed,
        search_kwargs=dict(
            pop_size=args.popsize,
            selection=args.selection,
            evaluation_timeout=args.timeout,
            memory_limit=args.memory * Gb,
            early_stop=args.early_stop,
        ),
    )
    fitness_fn = make_fn(args, train_dict, test_dict, feature_set, score_fn)
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def setup_mlp(args, train_dict, test_dict, feature_set, score_fn):
    hidden_size = get_hidden_size(train_dict, feature_set)
    classifier = MLPClassifier(lr=args.lr)
    classifier.initialize(hidden_size, device=GPU_DEVICE)
    fitness_fn = make_mlp_fn(
        args, train_dict, test_dict, feature_set, score_fn
    )
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def setup_pipeline(args, train_dict, test_dict, feature_set, score_fn):
    setup_fn = setup_pe_pipeline
    if args.pipeline == "mlp" and not args.autogoal:
        setup_fn = setup_mlp
    elif args.pipeline != "mlp" and args.autogoal:
        setup_fn = setup_automl

    return setup_fn(args, train_dict, test_dict, feature_set, score_fn)


def get_fitness_fn(args, train_dict, test_dict, feature_set, score_fn):
    maker = make_fn
    if args.pipeline == "mlp":
        maker = make_mlp_fn
    elif args.balanced:
        maker = make_balanced_fn

    return maker(args, train_dict, test_dict, feature_set, score_fn)
