FROM nvcr.io/nvidia/tensorflow:20.02-tf2-py3
MAINTAINER Guillermo Echegoyen <gblanco@lsi.uned.es>

WORKDIR /workspace
COPY scripts /workspace/scripts
COPY requirements.txt  /workspace
COPY transformers  /workspace/transformers

RUN cd /workspace && ./scripts/install_packages.sh

