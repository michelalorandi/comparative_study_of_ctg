#!/bin/bash

cd  ./scripts/src/models/discup

echo "Create Environment..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name discup_env
conda create -n discup_env python=3.8 && conda activate discup_env
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

echo "Environment is Ready!"
