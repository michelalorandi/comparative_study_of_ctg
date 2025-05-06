import os
import json
import random
import argparse
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch, load_control_values

import torch
from transformers import AutoTokenizer, CTRLLMHeadModel

CONTROL_ATTRIBUTES = {
    "World": [], 
    "Sports": ["Fitness"], 
    "Business": ["Finance", "Legal"], 
    "Science/Technology": ["Science", "Computing", "Technology"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_ctrl_model(hyperparameters: dict) -> Tuple[AutoTokenizer, CTRLLMHeadModel]:
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters['tokenizer'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = CTRLLMHeadModel.from_pretrained(hyperparameters['model']).to(device)
    return tokenizer, model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def transform_dataset_to_prompt_ctrl(examples: dict, args: argparse.Namespace) -> dict:
    control_values = load_control_values(args.control_attribute)

    transf_data = {
        "original_id": [],
        "prompt": [],
        "control_attribute": [],
        "control_attribute_value": [],
        "control_attribute_model_value": []
    }
    for index in range(len(examples['prompt'])):
        for value in control_values:
            model_values = CONTROL_ATTRIBUTES[value]

            if len(model_values) > 0:
                actual_value = model_values[0] if len(model_values) == 1 else random.choice(model_values)
                current_prompt = f'{actual_value} {examples["prompt"][index]}'
                transf_data['original_id'].append(examples['original_id'][index])
                transf_data['prompt'].append(current_prompt)
                transf_data['control_attribute'].append(args.control_attribute)
                transf_data['control_attribute_model_value'].append(actual_value)
                transf_data['control_attribute_value'].append(value)
    return transf_data


def generate_ctrl_sequence_batch(prompts: List[str], tokenizer: AutoTokenizer, model: CTRLLMHeadModel, hyperparameters: dict) -> List[str]:
    #print(prompts)
    #print(len(prompts))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    # assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()
    sequence_ids = model.generate(inputs["input_ids"], max_new_tokens=hyperparameters['max_new_tokens'])
    sequences = tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)
    return sequences


def execute_experiment(args: argparse.Namespace) -> None:
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    data_df = dataset_hf.map(transform_dataset_to_prompt_ctrl, batched=True, 
                              remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                              fn_kwargs={'args': args})
    print('Prompts Dataset length:', len(data_df['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['max_new_tokens'] = args.max_length

    tokenizer, model = init_ctrl_model(hyperparameters)

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

        texts = generate_ctrl_sequence_batch(batch['prompt'], tokenizer, model, hyperparameters)

        results['results'] += get_results_object_batch(batch['original_id'], batch['prompt'],
                                                       batch['control_attribute_value'], texts,
                                                       control_attribute_model_values=batch['control_attribute_model_value'])

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
                        choices=["topic"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=[None])
    arguments = parser.parse_args()

    arguments.batch_size = 3

    execute_experiment(arguments)

