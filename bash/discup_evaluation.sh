#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate csctg_eval_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('pplm_prompts' 'sts_benchmark_test' 'cloze_2018_test' 'owt_neutral_prompts')
len=20
attributes=('sentiment')
model='discup'

echo "Start Evaluation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            folder="${model}-${dataset}-${attribute}-None-len${len}-${seed}"
            python3 ./scripts/evaluation_pipeline.py --results_folder_path ./results --control_attribute "$attribute" --batch_size 128 --folder "$folder"
        done
    done
done
echo "End Evaluation."
