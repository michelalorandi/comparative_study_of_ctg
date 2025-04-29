#!/bin/bash

cd  ./scripts/src/models/bolt

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name bolt_env
conda create -n bolt_env python=3.8 && conda activate bolt_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.36.2
pip install datasets==2.16.1
echo "Environment created."
