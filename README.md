# QuAIL Experiments

Experiments conducted at UNED with QuAIL dataset.

# Reproduce

We use (dvc)[https://dvc.org/] to make our experiments reproducible.

Setup:

```bash
git clone https://github.com/m0n0l0c0/quail-experiments
cd quail-experiments
# setup your remote, any local dir will do really
mkdir ~/dvc_cache
dvc config --local core.remote local_dvc
dvc remote add --local -d local_dvc "~/dvc_cache"
```

And reproduce the end of the pipeline
```bash
dvc repro
```

# License

MIT License
