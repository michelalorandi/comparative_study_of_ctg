#!/bin/bash

echo "Create Environment..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda remove --name multi_ctg_env
conda env create -f ./scripts/src/models/multi_ctg/requirements.yml
conda activate multi_ctg_env

echo "Environment is Ready!"
