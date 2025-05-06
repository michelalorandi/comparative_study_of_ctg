#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Custom imports for utilities
from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch, load_control_values

# Model imports
import torch
from transformers import GPT2Tokenizer
from src.models.cat_paw.modeling_gpt2 import GPT2LMHeadModel

from src.models.cat_paw.pplm import latent_perturb


device = "cuda" if torch.cuda.is_available() else "cpu"

TOPIC = {
    "World": [], 
    "Sports": [], 
    "Business": [], 
    "Science/Technology": ["science", "space", "computers"]
}

SENTIMENT = {
    "positive": {
        "name": "very_positive",
        "class_label": 2
        },
    "negative": {
        "name": "very_negative",
        "class_label": 3
        }
}


def model_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-of-words', '-B', type=str, default=None, 
                        help='Bags of words used for PPLM-BoW. Multiple BoWs separated by ;')
    parser.add_argument('--discrim', '-D', type=str, default=None, 
                        choices=('clickbait', 'sentiment', 'toxicity'), 
                        help='Discriminator to use for loss-type 2')
    parser.add_argument('--label-class', type=int, default=-1, help='Class label used for the discriminator')

    args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(args):
    enc = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained(args['model_path'])
    model.to(device)
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False
    return model, enc


def generate_batch(model, enc, prompts, params):
    seq = [([50256] + enc.encode(tmp_text)) for tmp_text in prompts]
    bag_of_words = [params['bag_of_words']]
    
    current_index = 0 
    for tmp_bow in bag_of_words:
        params['bag_of_words'] = tmp_bow
        res = []
        for out in seq:
            context_del = len(out)
            if params['require_origin']:
                out1, out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(enc=enc, model=model, params=params, context=out,
                                                                        sample=params['sample'], device=device)
            else:
                out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(enc=enc, model=model, params=params, context=out,
                                                                        sample=params['sample'], device=device)



            if params['require_origin']:
                text_whole = enc.decode(out1.tolist()[0])

            out_perturb_copy = out_perturb

            generated = 0
            for out_perturb in out_perturb_copy:
                try:
                    text_whole = enc.decode(out_perturb.tolist()[0])
                    res.append(text_whole)
                except:
                    pass
                #collect_gen[current_index] = [out, out_perturb, out1]
                # Save the prefix, perturbed seq, original seq for each index

                current_index = current_index + 1
    return res


def execute_experiment(args: argparse.Namespace) -> None:
    control_values = load_control_values(args.control_attribute)

    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Prompts Dataset length:', len(dataset_hf['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['length'] = args.max_length

    model, tokenizer = init_model(hyperparameters)

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
        for index in tqdm(range(0, len(dataset_hf['train']), args.batch_size),
                        desc=f"Executing {experiment_name}"):
            
            model_values = TOPIC[control_val] if args.control_attribute == 'topic' else [SENTIMENT[control_val]["name"]]
            if len(model_values) > 0:
                actual_value = model_values[0] if len(model_values) == 1 else random.choice(model_values)

                hyperparameters['discrim'] = args.control_attribute if args.control_attribute == 'sentiment' else None
                hyperparameters['label_class'] = SENTIMENT[control_val]["class_label"] if args.control_attribute == 'sentiment' else None
                hyperparameters['bag_of_words'] = actual_value if args.control_attribute == 'topic' else None

                batch = dataset_hf['train'][index:index+args.batch_size]

                texts = generate_batch(model, tokenizer, batch['prompt'], hyperparameters)

                results['results'] += get_results_object_batch(batch['original_id'], batch['prompt'], 
                                                               [control_val for _ in range(len(batch['prompt']))], 
                                                               texts, control_attribute_model_values=[actual_value for _ in range(len(batch['prompt']))])

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
                        choices=["sentiment", "topic"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
#CUDA_VISIBLE_DEVICES=0 python pplm.py -B military --cond-text "The potato" --length 50 --gamma 1.5 --num-iterations 3 --num-samples 10 --stepsize 0.03 --window-length 5 --fusion-kl-scale 0.01 --fusion-gm-scale 0.99 --sample
