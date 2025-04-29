#!/bin/bash

echo "Create Environment..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name dexperts_env
conda env create -f ./scripts/src/models/dexperts/requirements.yml
conda activate dexperts_env

echo "Environment is Ready!"
