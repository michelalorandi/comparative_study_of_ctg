#!/bin/bash

echo "Create Environment..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda env remove --name prior_ctg
conda env create -f ./scripts/src/models/prior_control/requirements.yml
conda activate prior_ctg

echo "Environment is Ready!"
