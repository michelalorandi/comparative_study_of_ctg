#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate ctg_bart_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('yelp_review' 'one-billion-words')
lenghts=(20)

echo "Start Generation..."
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for len in "${lenghts[@]}"
        do
            python3 ./scripts/ctg_bart_repro_generation_pipeline.py --results_folder_path ./results --seed "$seed" --control_attribute keywords --dataset "$dataset" --batch_size 128 --model ctg_bart --max_length "$len"
        done
    done
done
echo "End Generation."
