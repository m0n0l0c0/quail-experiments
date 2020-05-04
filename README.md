# QuAIL Experiments

Experiments conducted at UNED with QuAIL dataset.

# Reproduce

We use (dvc)[https://dvc.org/] to make our experiments reproducible. To do so,
issue (dvc required):

```bash
git clone https://github.com/m0n0l0c0/quail-experiments
cd quail-experiments
dvc repro ./stages/preparation/transform_quail_train_to_race.dvc
```

By now, the last step in the pipeline is `transform_quail_train_to_race`, it
downloads data from quail and transforms quail data to a compiled race like dataset.

# License

MIT License
