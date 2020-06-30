#!/bin/bash

output_dir="data/quail"

[[ ! -d "${output_dir}" ]] && mkdir -p "${output_dir}"

quail_urls=(
  "https://worksheets.codalab.org/rest/bundles/0x33ab04f4e1014ef591f6b81df092fe55/contents/blob/" "dev_answers.json"
  "https://worksheets.codalab.org/rest/bundles/0x3f22902d168b4e79b2fc2b1cba7e7201/contents/blob/" "train.json"
  "https://worksheets.codalab.org/rest/bundles/0xf868182531244692878454c0f949822e/contents/blob/" "train_answers.json"
  "https://worksheets.codalab.org/rest/bundles/0x7e113f2e5c7d4cb583e21224770880c0/contents/blob/" "dev.json"
  "https://worksheets.codalab.org/rest/bundles/0x051eedf8b8904bebbedbccce5430d5a1/contents/blob/" "eval.py"
)

for (( i = 0; i < "${#quail_urls[@]}"; i+=2 )); do
  wget -q --show-progress "${quail_urls[$i]}" -O "${output_dir}/${quail_urls[$(( $i +1 ))]}" 
done
