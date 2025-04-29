#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate catpaw_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('data/pplm_prompts.csv' 'data/owt_neutral_prompts.csv' 'data/cloze_2018_test.csv' 'data/sts_benchmark_test.csv')
lenghts=(50)
attributes=('sentiment' 'topic')

echo "Start Generation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            for len in "${lenghts[@]}"
            do
                python3 ./scripts/cat_paw_generation_pipeline.py --results_folder_path ./results --seed "$seed" --control_attribute "$attribute" --dataset_filepath "$dataset" --batch_size 128 --model cat_paw --max_length "$len"
            done
        done
    done
done
echo "End Generation."
