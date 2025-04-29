#!/bin/bash

cd  ./scripts/src/models/ctrl

echo "Environment creation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda create -n ctrl_env python=3.8 && conda activate ctrl_env
pip install -r requirements.txt
echo "Environment created."
