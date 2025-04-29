#!/bin/bash

echo "Environment activation..."
eval "$($(which conda) 'shell.bash' 'hook')"
conda activate llm_env
echo "Environment activated."

seeds=(789 3443 9817)
datasets=('data/pplm_prompts.csv' 'data/owt_neutral_prompts.csv' 'data/cloze_2018_test.csv' 'data/sts_benchmark_test.csv')
prompts=('zero_shot' 'few_shot')
attributes=('sentiment' 'topic' 'keywords' 'multiple')

echo "Start Generation..."
for attribute in "${attributes[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for prompt in "${prompts[@]}"
        do
            for seed in "${seeds[@]}"
            do
                python3 ./scripts/llm_generation_pipeline.py --tokens_path ./scripts/src/tokens/api_tokens.json --results_folder_path ./results --seed "$seed" --control_attribute "$attribute" --dataset_filepath "$dataset" --prompt_type "$prompt" --batch_size 128 --model meta-llama/Llama-2-70b-chat-hf --max_length 100
            done
        done
    done
done
echo "End Generation."