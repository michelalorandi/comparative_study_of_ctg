#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate csctg_eval_env
echo "Environment activated."
seeds=(789 3443 9817)
dataset='bookcorpus_prompts_'
nums=(419611 1458592 1719584 1867826 2341058 3744855 4108604 4614227 9149733 9906821)
len=100
attribute='sentiment'
model='gedi'

echo "Start Generation..."
for seed in "${seeds[@]}"
do
    for num in "${nums[@]}"
    do
        value="${dataset}${num}"
        folder="${model}-${value}-${attribute}-None-len${len}-${seed}"
        python3 ./scripts/evaluation_pipeline.py --results_folder_path ./results --control_attribute "$attribute" --batch_size 128 --folder "$folder"
    done
done
echo "End Generation."
