"""
 Script to execute the generation pipeline
"""
# standard libraries import
import os
from typing import List
from datetime import datetime

# non-standard libraries import
from tqdm import tqdm
from datasets import load_dataset

# my scripts import
from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch
from src.models.llms_testing import (transform_dataset_to_prompt, set_huggingface_token,
                              load_huggingface_model, llms_send_request_batch, transform_dataset_to_prompt,
                              transform_dataset_to_prompt_multiple)
# models related import
import torch


def load_transform_dataset(args):
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Original Dataset length:', len(dataset_hf['train']))
    if args.control_attribute == 'multiple':
        return dataset_hf.map(transform_dataset_to_prompt_multiple, batched=True,
                            remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                            fn_kwargs={'args': args})
    return dataset_hf.map(transform_dataset_to_prompt, batched=True,
                            remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                            fn_kwargs={'args': args})


def execute_experiment(args):
    data_df = load_transform_dataset(args)
    print('Prompts Dataset length:', len(data_df['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['max_new_tokens'] = args.max_length

    model, tokenizer = load_huggingface_model(args.model, args.seed)

    # create results directory for the current experiment if it doesn't exist
    check_folder_exists_and_create(args.results_folder_path)
    check_folder_exists_and_create(os.path.join(args.results_folder_path, args.control_attribute))
    res_dir = os.path.join(args.results_folder_path, args.control_attribute, experiment_name)
    check_folder_exists_and_create(res_dir)

    results = {
        'start_time_experiment': get_current_date(),
        'experimental_settings': {
            "task": get_task(args.dataset_filepath),
            "dataset_filepath": args.dataset_filepath,
            "prompt": args.prompt_type,
            "seed": args.seed,
            "control_attribute": args.control_attribute,
            "model": args.model
        },
        'model_hyperparameters': hyperparameters,
        'results': []
    }
    results_filename = os.path.join(res_dir, f'raw_{experiment_name}.json')
    for index in tqdm(range(0, len(data_df['train']), args.batch_size),
                      desc=f"Executing {experiment_name}"):
        batch = data_df['train'][index:index+args.batch_size]

        params = {
            'hyperparameters': hyperparameters,
            'model': args.model,
            'seed': args.seed
        }

        _, texts = llms_send_request_batch(batch['prompt'], params, model, tokenizer)

        results['results'] += get_results_object_batch(batch['original_id'], batch['prompt'],
                                                       batch['control_attribute_value'], texts)

        save_json(results, results_filename)

    results['end_time_experiment'] = get_current_date()

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_path', metavar='path',
                        default=os.path.join('.', 'scripts', 'src', 'tokens', 'api_tokens.json'))
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--control_attribute', type=str, required=True,
                        choices=["sentiment", "topic", "keywords", "multiple"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    # Load your API key from an environment variable or secret management service
    if 'falcon' in arguments.model or 'llama' in arguments.model:
        set_huggingface_token(arguments.tokens_path)

    execute_experiment(arguments)
