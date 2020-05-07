#!/bin/bash

### 
# Try to doanload a model from huggingface. This is by now means exhaustive.
###

model=${1:-"bert-base-uncased"}; shift
model_dir=${1:-$model}; shift

urls=(
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-pytorch_model.bin"
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-config.json"
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-modelcard.json"
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-rust_model.ot"
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-tf_model.h5"
  "https://s3.amazonaws.com/models.huggingface.co/bert/${model}-vocab.txt"
)

[[ ! -d $model_dir ]] && mkdir -p $model_dir

echo "${urls[@]}" | xargs -n 1 -P 8 \
  wget -q --show-progress -O ${model_dir}/$(echo {} | tr '-' ' ' | awk '{print $NF}') 
