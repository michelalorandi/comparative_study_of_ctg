#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import argparse

from tqdm import tqdm
from datasets import load_dataset

# Custom imports for utilities
from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch, load_control_values

# Model imports
import torch

from src.models.dexperts.dexperts_generation import DExpertsGeneration



def init_model(args: dict, control_attribute_value: str, seed: int) -> DExpertsGeneration:
    if control_attribute_value == 'positive':
        expert_name_or_path = args['pos_model']
        antiexpert_name_or_path = args['neg_model']
    else:
        expert_name_or_path = args['neg_model']
        antiexpert_name_or_path = args['pos_model']

    model = DExpertsGeneration(
        base_model=args['model'], 
        expert_model=expert_name_or_path,
        antiexpert_model=antiexpert_name_or_path,
        seed=seed
    )
    return model


def execute_experiment(args: argparse.Namespace) -> None:
    control_values = load_control_values(args.control_attribute)

    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Prompts Dataset length:', len(dataset_hf['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_tokens}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['max_tokens'] = args.max_tokens
    model = init_model(hyperparameters, 'positive', args.seed)

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
    for control_val in control_values:
        if control_val == 'negative':
            tmp = model.expert
            model.expert = model.antiexpert
            model.antiexpert = tmp
        for index in tqdm(range(0, len(dataset_hf['train']), args.batch_size),
                        desc=f"Executing {experiment_name}"):

            batch = dataset_hf['train'][index:index+args.batch_size]

            texts = model.generate(prompt=batch['prompt'], max_len=args.max_tokens, filter_p=hyperparameters['filter_p'],
                                   p=hyperparameters['p'], alpha=hyperparameters['alpha'])
            texts = [batch['prompt'][i] + texts[i] for i in range(len(texts))]

            results['results'] += get_results_object_batch(batch['original_id'], batch['prompt'], 
                                                            [control_val for _ in range(len(batch['prompt']))], 
                                                            texts)

            save_json(results, results_filename)

    results['end_time_experiment'] = get_current_date()

    if model is not None:
        del model
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
                        choices=["sentiment", "topic"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    execute_experiment(arguments)
