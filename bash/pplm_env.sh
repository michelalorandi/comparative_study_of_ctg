#!/bin/bash

cd  ./scripts/src/models/pplm

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name pplm_env
conda create -n critic_env python=3.8 && conda activate critic_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install datasets
echo "Environment created."
