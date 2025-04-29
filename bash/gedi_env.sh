#!/bin/bash

echo "Create Environment..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name gedi_env
conda env create -f ./scripts/src/models/gedi/requirements.yml
conda activate gedi_env

echo "Environment is Ready!"
