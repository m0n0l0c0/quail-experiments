#!/bin/bash

if [[ "$#" -lt 1 ]]; then
  echo "dump_metrics.sh <outfile>"
  exit 1
fi

outfile=$1

[[ ! -d $(dirname $outfile) ]] && mkdir -p $(dirname $outfile)

dvc metrics show --show-json data/metrics/bert/generic/*/*_epochs_2/logits_evaluation.json > $outfile

