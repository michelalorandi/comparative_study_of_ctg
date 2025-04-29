#!/bin/bash

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name ctg_eval_env
conda create -n ctg_eval_env python=3.8 && conda activate ctg_eval_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
pip install datasets
pip install sentencepiece==0.1.96
pip install accelerate==0.26.1
pip install evaluate==0.4.1
pip install spacy==3.7.3
pip install lemminflect==0.2.3
pip install scikit-learn
python3 -m spacy download en_core_web_sm
echo "Environment created."
