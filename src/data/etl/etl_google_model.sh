#!/bin/bash

if [[ "$#" -lt 1 ]]; then
  echo "Usage setup_google_model.sh <model_name> <url> [<zip_file>, <force>]"
  echo "  (hint: provide zip_file when last segment of url doesn't match the actual filename)"
  exit 0
fi

ch_to_project_dir(){
  # chdir to project root
  scriptdir=$(dirname -- "$(realpath -- "$0")")
  rootdir=$(echo $scriptdir | sed -e 's/\(quail-experiments\).*/\1/')
  cwd=$(pwd)
  cd $rootdir >/dev/null
}

ensure_dirs(){
  for i in "$@"; do
    [[ ! -d $i ]] && mkdir -p $i
  done
}

maybe_download_data(){
  local file_path=$1; shift
  local url=$1; shift
  local force=${1:-0}
  local download=0
  if [[ -f "${file_path}" && "${force}" -eq 1 ]]; then
    download=1
  elif [[ ! -f "${file_path}" ]]; then
    download=1
  fi
  if [[ "${download}" -eq 1 ]]; then
    wget -q --show-progress -O $file_path $url
  fi
}

maybe_extract_data(){
  local zip_path=$1; shift
  local end_path=$1; shift
  local force=${1:-0}; shift
  local move=0
  if [[ -f "${end_path}" && "${force}" -eq 1 ]]; then
    move=1
  elif [[ ! -f "${end_path}" ]]; then
    move=1
  fi
  if [[ "${move}" -eq 1 ]]; then
    # ensure it does not exist so mv doesn't try to move it inside
    rm -rf $end_path
    unzip $zip_path -d /tmp/
    zip_file=$(basename $zip_path)
    mv "/tmp/${zip_file%.*}" "${end_path}"
  fi
}

ch_to_project_dir

model_name=$1; shift
url=$1; shift
# if not provided, try to guess the file name by the last segment of the url,
# another would be to accept base url and resource access by zip name
model_zip_file=${1:-$(echo $url | awk 'BEGIN { FS = "/" }; {print $NF}')}; shift
force=${1:-0}

## data directories
data_dir="data"
raw_dir="${data_dir}/raw"
all_models_dir="${data_dir}/models"

model_zip_path="${raw_dir}/${model_zip_file}"
model_dir="${all_models_dir}/${model_name}"

convert_script="src/data/etl/convert_bert_original_tf_checkpoint_to_pytorch.py"

# (only endpoints, -p option enabled)
ensure_dirs $raw_dir $all_models_dir

# download, unzip, rename, transform
maybe_download_data $model_zip_path $url $force
maybe_extract_data $model_zip_path $model_dir $force
python3 $convert_script \
  --tf_checkpoint_path "$(find $model_dir -type f -iname '*_model.ckpt.data*' | sed -e 's/\(.*_model.ckpt\).*/\1/')" \
  --bert_config_file "$(find $model_dir -type f -iname '*_config.json')" \
  --pytorch_dump_path "${model_dir}/pytorch_model.bin"

# move bert_model.ckpt. .. -> model.ckpt
for file in $(find "${model_dir}" -iname 'bert_*'); do
  name=$(basename $file)
  no_bert_name=$(echo "${name}" | sed -e 's/bert_\(.*\)/\1/')
  mv "${model_dir}/${name}" "${model_dir}/${no_bert_name}"
done
