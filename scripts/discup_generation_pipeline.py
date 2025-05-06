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

from src.models.discup.control_generation import  CTG

from src.models.discup.utils import seed_everything


def execute_experiment(args: argparse.Namespace) -> None:
    control_values = load_control_values(args.control_attribute)

    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Prompts Dataset length:', len(dataset_hf['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{(args.max_length-100)}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_name_or_path = hyperparameters['model_name_or_path']
    args.embedding_checkpoint = hyperparameters['embedding_checkpoint']
    args.pseudo_token = hyperparameters['pseudo_token']
    args.temperature = hyperparameters['temperature']
    args.beta = hyperparameters['beta']
    args.tuning_name = hyperparameters['tuning_name']
    args.use_lm_finetune = hyperparameters['use_lm_finetune']
    args.lstm_dropout = hyperparameters['lstm_dropout']
    args.disc_embedding_checkpoint = hyperparameters['disc_embedding_checkpoint']
    args.prompt_pad_length = hyperparameters['prompt_pad_length']
    args.ranking_scope = hyperparameters['ranking_scope']
    args.max_prompt_length = hyperparameters['max_prompt_length']

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
    for control_val in tqdm(control_values, desc=f"Executing {experiment_name}"):
        args.prompt_type = 'neutral'
        args.target_type = control_val
        args.embedding_checkpoint = hyperparameters['embedding_checkpoint'][control_val]
        args.template = eval(hyperparameters['template'][control_val]) if type(hyperparameters['template'][control_val]) is not tuple else hyperparameters['template'][control_val]

        model = CTG(args, dataset_hf['train']['prompt'])
        texts = model.test()

        results['results'] += get_results_object_batch(dataset_hf['train']['original_id'], dataset_hf['train']['prompt'], 
                                                        [control_val for _ in range(len(dataset_hf['train']['prompt']))], 
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
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    seed_everything(arguments.seed)

    execute_experiment(arguments)
