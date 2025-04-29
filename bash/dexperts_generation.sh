#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate dexperts_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('data/pplm_prompts.csv' 'data/owt_neutral_prompts.csv' 'data/cloze_2018_test.csv' 'data/sts_benchmark_test.csv')
lenghts=(20)
attributes=('sentiment')

echo "Start Generation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            for len in "${lenghts[@]}"
            do
                python3 ./scripts/dexperts_generation_pipeline.py --results_folder_path ./results --seed "$seed" --control_attribute "$attribute" --dataset_filepath "$dataset" --batch_size 64 --model dexperts --max_tokens "$len"
            done
        done
    done
done
echo "End Generation."
