#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate csctg_eval_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('pplm_prompts' 'sts_benchmark_test' 'cloze_2018_test' 'owt_neutral_prompts')
prompts=('zero_shot' 'few_shot')
len=100
attributes=('sentiment' 'topic' 'keywords' 'multiple')
model='falcon-40b-instruct'

echo "Start Evaluation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            for prompt in "${prompts[@]}"
            do
                folder="${model}-${dataset}-${attribute}-${prompt}-len${len}-${seed}"
                python3 ./scripts/evaluation_pipeline.py --results_folder_path ./results --control_attribute "$attribute" --batch_size 128 --folder "$folder"
            done
        done
    done
done
echo "End Evaluation."
