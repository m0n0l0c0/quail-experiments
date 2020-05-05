#!/bin/bash

scriptdir=$(dirname -- "$(realpath -- "$0")")
rootdir=$(dirname $scriptdir)
cd $rootdir >/dev/null

# dockerize by default
dockerize=${1:-1}

# download some repos we want locally, installed in docker
git clone https://github.com/m0n0l0c0/transformers
git clone https://github.com/m0n0l0c0/mc-transformers src/mc-transformers
if [[ "$dockerize" -eq 0 ]]; then
  ./install_packages.sh $dockerize
else
  docker build -t quail-experiments .
fi

cd - >/dev/null
