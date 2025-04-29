#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate csctg_eval_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('yelp_review' 'one-billion-words')
len=20
attribute='keywords'
model='ctg_bart'

echo "Start Generation..."
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        folder="${model}-${dataset}-${attribute}-None-len${len}-${seed}"
        python3 ./scripts/evaluation_pipeline.py --results_folder_path ./results --control_attribute "$attribute" --batch_size 128 --folder "$folder"
    done
done
echo "End Generation."
