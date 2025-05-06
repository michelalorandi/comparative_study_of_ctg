#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import argparse

from typing import List
from tqdm import tqdm
from datasets import load_dataset

# Custom imports for utilities
from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch, load_control_values

# Model imports
import tensorflow as tf

import src.models.non_res_prompt.Utils as Utils
from src.models.non_res_prompt.DecodingStrategies import WordInclusion
from src.models.non_res_prompt.NonResidualAttention import InferenceUtils


def set_seed(seed):
    tf.random.set_seed(seed)



def generate_text(model, tokenizer, promptGenerator, prompt: str, max_length: int, keywords: List[str], num_beams: int) -> str:
    prompt_len = len(prompt.split())
    promptIDs = [promptGenerator.createPromptIDs(' '.join(words), prompt_len) for words in keywords]
    context_len = len(tokenizer.encode(prompt))
    WordInclusion.setCurrentInclusionWords(keywords, max_length, contextLen=context_len, numBeams=num_beams)

    sample_contexts = [prompt for _ in keywords]
    result = InferenceUtils.batchGeneration(model, tokenizer, sample_contexts, promptIDs, max_length, num_beams)

    return result[0]


def generate_texts_batch(model, tokenizer, promptGenerator, prompts: List[str], max_length: int, keywords: List[str], num_beams: int) -> List[str]:
    texts = []

    for prompt in prompts:
        texts.append(generate_text(model, tokenizer, promptGenerator, prompt, max_length, keywords, num_beams))

    return texts


def execute_experiment(args: argparse.Namespace) -> None:
    control_values = load_control_values(args.control_attribute)

    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Prompts Dataset length:', len(dataset_hf['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)

    # create results directory for the current experiment if it doesn't exist
    check_folder_exists_and_create(args.results_folder_path)
    check_folder_exists_and_create(os.path.join(args.results_folder_path, args.control_attribute))
    res_dir = os.path.join(args.results_folder_path, args.control_attribute, experiment_name)
    check_folder_exists_and_create(res_dir)

    model, tokenizer, maxSeqLen, promptGenerator = Utils.loadPaperGPT2LargeModel()

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
        for index in tqdm(range(0, len(dataset_hf['train']), args.batch_size),
                        desc=f"Executing {experiment_name}"):
            
            batch = dataset_hf['train'][index:index+args.batch_size]
            texts = generate_texts_batch(model, tokenizer, promptGenerator, batch['prompt'], args.max_length, control_val, hyperparameters['num_beams'])

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
                        choices=["keywords"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=[None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
