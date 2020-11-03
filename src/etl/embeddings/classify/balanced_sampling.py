import numpy as np

from classification import get_x_y_from_dict
from imblearn.under_sampling import RandomUnderSampler


# raw resampling, X_train and y_train must be given
def balanced_resample(seed=42, **kwargs):
    X_train, y_train = kwargs["X_train"], kwargs["y_train"]
    sampler = RandomUnderSampler(random_state=seed)
    return sampler.fit_resample(X_train, y_train)


# resamples maintaining shape and casting to float
def balanced_resample_for_training(train_dict, feature_set, seed, **kwargs):
    X_train, y_train = get_x_y_from_dict(
        train_dict,
        features=feature_set,
        **kwargs
    )
    orig_shape = None
    # prepare input for under sampler
    if len(X_train.shape) > 0:
        orig_shape = list(X_train.shape)
        X_train = X_train.reshape(orig_shape[0], -1)

    X_train, y_train = balanced_resample(
        seed=seed, X_train=X_train, y_train=y_train
    )

    if orig_shape is not None:
        X_train = X_train.reshape([X_train.shape[0], *orig_shape[1:]])

    # ensure everything is floating...
    return dict(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
    )


def balanced_sampling_iter(epochs, train_dict, features, **kwargs):
    for ep in range(epochs):
        data_dict = balanced_resample_for_training(
            train_dict, features, seed=ep
        )
        yield (
            data_dict["X_train"], data_dict["y_train"],
        )
