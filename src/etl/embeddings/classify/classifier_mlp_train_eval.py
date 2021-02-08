import numpy as np

from sklearn.metrics import classification_report

from dataset_class import Dataset, EmbeddingsDataset
from dataset import get_x_y_from_dict, get_dataset_class_proportions
from balanced_sampling import balanced_sampling_iter
from torch.utils.data.dataloader import DataLoader


# Deprecated
def std_train_classifier(
    classifier,
    train_epochs,
    train_dict,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
):
    from classifier_setup import GPU_DEVICE
    dataset_rounds = get_dataset_class_proportions(train_dict)
    if not classifier.is_initialized:
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier.initialize(hidden_size, device=GPU_DEVICE)

    classifier.set_score_fn(score_fn)
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
                score_fn=score_fn,
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


# Deprecated
def std_eval_classifier(
    classifier,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
    return_y=False,
):
    y_preds_list = None
    classifier.model.eval()
    X_test, y_test = get_x_y_from_dict(
        test_dict, features=feature_set, cast=np.float32
    )
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


def scatter_train_classifier(
    classifier,
    train_epochs,
    train_dict,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
):
    from classifier_setup import GPU_DEVICE
    if not classifier.is_initialized:
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier.initialize(hidden_size, device=GPU_DEVICE)

    classifier.set_score_fn(score_fn)
    train_dict.prepare_for_train(feature_set)
    for epoch in range(1, train_epochs + 1):
        classifier.model.train()
        epoch_loss = 0
        epoch_acc = 0

        train_loader = DataLoader(
            dataset=train_dict,
            batch_size=batch_size,
            shuffle=True
        )
        for batch in train_loader:
            X_train, y_train = batch
            loss, acc = classifier.fit(X_train, y_train)
            epoch_loss += loss
            epoch_acc += acc

        if print_result:
            eval_acc, y_test, y_preds = scatter_eval_classifier(
                classifier,
                test_dict,
                feature_set,
                batch_size,
                score_fn=score_fn,
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


def scatter_eval_classifier(
    classifier,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
    return_y=False,
):
    y_preds_list = None
    classifier.model.eval()
    X_test, y_test = test_dict.get_x_y_from_dict(features=feature_set)
    X_test.prepare_for_eval(feature_set)
    test_loader = DataLoader(
        dataset=X_test,
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
    return y_test.to_list(), y_preds_list


def train_classifier(classifier, **kwargs):
    if isinstance(kwargs["train_dict"], Dataset):
        train_fn = scatter_train_classifier
    else:
        train_fn = std_train_classifier

    return train_fn(classifier, **kwargs)


def eval_classifier(classifier, **kwargs):
    if isinstance(kwargs["test_dict"], Dataset):
        eval_fn = scatter_eval_classifier
    else:
        eval_fn = std_eval_classifier

    return eval_fn(classifier, **kwargs)


def get_hidden_size(train_dict, features=None):
    if isinstance(train_dict, Dataset):
        sample = train_dict.first(x=True, y=False)
        return sample.shape[-1]
    else:
        X_train, _ = get_x_y_from_dict(train_dict, features=features)
        return X_train.reshape(X_train.shape[0], -1).shape[-1]
