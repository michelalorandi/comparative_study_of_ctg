#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate gedi_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('data/bookcorpus_prompts_')
nums=(419611 1458592 1719584 1867826 2341058 3744855 4108604 4614227 9149733 9906821)

echo "Start Generation..."
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for num in "${nums[@]}"
        do
            value="${dataset}${num}.csv"
            python3 ./scripts/gedi_generation_pipeline.py --results_folder_path ./results --seed "$seed" --control_attribute sentiment --dataset_filepath "$value" --batch_size 128 --max_tokens 100 --model gedi
        done
    done
done
echo "End Generation."
