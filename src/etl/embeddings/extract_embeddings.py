import os
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from transformers import is_tf_available, Trainer
from mc_transformers.utils_mc import MultipleChoiceDataset, Split
from mc_transformers.data_classes import DataCollatorWithIds
from mc_transformers.mc_transformers import (
    compute_metrics,
    setup,
    softmax
)

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
        "-s", "--split", required=False, default="dev", type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to extract embeddings from"
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
        "-o", "--output_dir", required=False, type=str,
        help="Output directory to store predictions and embeddings"
    )
    parser.add_argument(
        "--single_items", action="store_true",
        help="Whether to store the embeddings and logits as separate files"
    )
    return parser.parse_args()


def mc_setup(args_file, split):
    config_kwargs = {
        "return_dict": True,
        "output_hidden_states": True,
    }
    all_args, processor, config, tokenizer, model = setup(
        [args_file], config_kwargs=config_kwargs
    )
    model_args, data_args, dir_args, training_args, window_args = (
        all_args.values()
    )
    data_collator = DataCollatorWithIds()
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split(split),
            enable_windowing=window_args.enable_windowing,
            stride=window_args.stride,
            no_answer_text=window_args.no_answer_text,
        )
        if training_args.do_eval
        else None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator.collate,
    )
    return all_args, model, trainer, eval_dataset


def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def get_model_results(logits, labels):
    preds = np.argmax(softmax(logits, axis=1), axis=1)
    return (preds == labels)


def save_data(output_dir, prefix, single_items, **kwargs):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if single_items:
        for key, value in kwargs.items():
            fpath = os.path.join(output_dir, f"{prefix}_{key}.pt")
            with open(fpath, "wb") as fout:
                pickle.dump(value, fout)

    fpath = os.path.join(output_dir, f"{prefix}_data.pt")
    with open(fpath, "wb"):
        pickle.dump(kwargs, fout)


def embed_dataset(model, dataloader, device):
    embeddings = None
    logits = None
    labels = None

    max_pooling = torch.nn.MaxPool1d(model.config.hidden_size)
    model = model.to(device)

    for inputs in tqdm(dataloader, desc="Embedding"):
        inputs = prepare_inputs(inputs, device)
        with torch.no_grad():
            output = model(**inputs)
            last_hidden_state = output.hidden_states[-1]
            pooled_output = last_hidden_state
            # pooled_output = max_pooling(last_hidden_state)

            numpyfied_logits = output.logits.cpu().numpy()
            numpyfied_output = pooled_output.cpu().numpy()
            numpyfied_labels = None

            if "labels" in inputs:
                numpyfied_labels = inputs["labels"].cpu().numpy()

            if embeddings is None:
                embeddings = np.array(numpyfied_output)
                logits = np.array(numpyfied_logits)
                if numpyfied_labels is not None:
                    labels = np.array(numpyfied_labels)
            else:
                embeddings = np.vstack([embeddings, numpyfied_output])
                logits = np.vstack([logits, numpyfied_logits])
                if numpyfied_labels is not None:
                    labels = np.hstack([labels, numpyfied_labels])

    return embeddings, logits, labels


def main(split, args_file, gpu, output_dir, single_items):
    all_args, model, trainer, eval_dataset = mc_setup(args_file, split)
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    device = torch.device("cuda", index=gpu)
    embeddings, logits, labels = embed_dataset(model, eval_dataloader, device)

    # labels should not be null
    num_samples = logits.shape[0]
    num_choices = logits.shape[1]
    pred_labels_correct = get_model_results(logits, labels).tolist()
    # 1 Correct / 0 Incorrect
    pred_labels_correct = np.array([int(p) for p in pred_labels_correct])
    embeddings = embeddings.reshape(
        num_samples, num_choices, *embeddings.shape[1:]
    )

    save_data(
        output_dir,
        split,
        single_items,
        embeddings=embeddings,
        labels=pred_labels_correct,
        logits=logits
    )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
