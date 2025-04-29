#!/bin/bash

cd  ./scripts/src/models/cat_paw

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name catpaw_env
conda create -n catpaw_env python=3.8 && conda activate catpaw_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
echo "Environment created."
