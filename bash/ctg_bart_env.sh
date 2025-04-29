#!/bin/bash

cd  ./scripts/src/models/ctg_bart

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda create -n ctg_bart_env python=3.8 && conda activate ctg_bart_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install datasets
echo "Environment created."
