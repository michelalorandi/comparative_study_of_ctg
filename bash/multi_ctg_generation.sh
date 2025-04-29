#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate multi_ctg_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('pplm_prompts' 'sts_benchmark_test' 'cloze_2018_test' 'owt_neutral_prompts')
attributes=('multiple')

echo "Start Generation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            python3 ./scripts/multi_ctg_generation_pipeline.py --results_folder_path ./results --seed "$seed" --control_attribute "$attribute" --dataset_filepath "$dataset" --batch_size 128 --model multi_ctg --max_length 50
        done
    done
done
echo "End Generation."
