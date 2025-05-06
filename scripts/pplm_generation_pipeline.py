#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import os
import random
import argparse
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, get_results_object_batch, load_control_values

import torch
from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
from src.models.pplm.pplm import full_text_generation, get_bag_of_words_indices, set_generic_model_params, DISCRIMINATOR_MODELS_PARAMS


TOPIC = {
    "World": [], 
    "Sports": [], 
    "Business": [], 
    "Science/Technology": ["science", "space", "technology"]
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



# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_pplm(hyperparameters: dict, control_attribute: str) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    pretrained_model = hyperparameters['pretrained_model']
    discrim = control_attribute if control_attribute == 'sentiment' else 'sentiment' if control_attribute == 'multiple' else None
    discrim_weights = hyperparameters['discrim_weights']
    discrim_meta = hyperparameters['discrim_meta']

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def run_pplm_example(
        model,
        tokenizer,
        cond_text,
        hyperparameters: dict,
        bag_of_words=None,
        discrim=None,
        class_label=None
):
    num_samples = hyperparameters['num_samples']
    length = hyperparameters['length']
    stepsize = hyperparameters['stepsize']
    temperature = hyperparameters['temperature']
    top_k = hyperparameters['top_k']
    sample = hyperparameters['sample']
    num_iterations = hyperparameters['num_iterations']
    grad_length = hyperparameters['grad_length']
    horizon_length = hyperparameters['horizon_length']
    window_length = hyperparameters['window_length']
    decay = hyperparameters['decay']
    gamma = hyperparameters['gamma']
    gm_scale = hyperparameters['gm_scale']
    kl_scale = hyperparameters['kl_scale']
    

    # figure out conditioning text
    raw_text = cond_text
    tokenized_cond_text = tokenizer.encode(
        tokenizer.bos_token + raw_text,
        add_special_tokens=False
    )

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    _, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale
    )

    generated_texts = []

    bow_word_ids = set()
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize perturbed text
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
            generated_texts.append(pert_gen_text)
        except:
            pass

    return generated_texts


def get_bag_of_words_discriminator_class(control_attribute: str, control_attribute_value: str or List[str], control_attribute_value_model: str = None):
    # return bag_of_words, discriminator, class_label
    if control_attribute == 'topic':
        return control_attribute_value_model, None, None
    elif control_attribute == 'sentiment':
        return None, control_attribute, SENTIMENT[control_attribute_value]["class_label"]
    elif control_attribute == 'multiple':
        return control_attribute_value_model[1], 'sentiment', SENTIMENT[control_attribute_value[0]]["class_label"]
    return None, None, None


def generate_text(batch: List[str], model, tokenizer, hyperparameters: dict, control_attribute: str, control_attribute_value: List[str] or List[List[str]], control_attribute_value_model: List[str]):
    generated_texts = []

    for i in range(len(batch)):
        bag_words, discrim, class_label = get_bag_of_words_discriminator_class(control_attribute, control_attribute_value[i], control_attribute_value_model[i])
        prompt = batch[i]
        texts = run_pplm_example(model, tokenizer, prompt, hyperparameters, bag_of_words=bag_words, discrim=discrim, class_label=class_label)
        generated_texts.append(texts[0])
    return generated_texts


def transform_dataset_to_prompt_pplm(examples: dict, args: argparse.Namespace) -> dict:
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
            model_values = TOPIC[value] if args.control_attribute == 'topic' else [SENTIMENT[value]["name"]]

            if len(model_values) > 0:
                actual_value = model_values[0] if len(model_values) == 1 else random.choice(model_values)
                current_prompt = f'{actual_value}:{examples["prompt"][index]}'
                transf_data['original_id'].append(examples['original_id'][index])
                transf_data['prompt'].append(current_prompt)
                transf_data['control_attribute'].append(args.control_attribute)
                transf_data['control_attribute_model_value'].append(actual_value)
                transf_data['control_attribute_value'].append(value)
    return transf_data


def transform_dataset_to_prompt_pplm_multiple(examples: dict, args: argparse.Namespace) -> dict:
    sent_control_vals = load_control_values("sentiment")
    topic_control_vals = load_control_values("topic")

    transf_data = {
        "original_id": [],
        "prompt": [],
        "control_attribute": [],
        "control_attribute_value": [],
        "control_attribute_model_value": []
    }
    for index in range(len(examples['prompt'])):
        for sent_val in sent_control_vals:
            for topic_val in topic_control_vals:
                sent_model_value = SENTIMENT[sent_val]["name"]
                topic_model_values = TOPIC[topic_val]

                if len(topic_model_values) > 0:
                    actual_topic_value = topic_model_values[0] if len(topic_model_values) == 1 else random.choice(topic_model_values)
                    current_prompt = f'{sent_model_value} {actual_topic_value}:{examples["prompt"][index]}'
                    transf_data['original_id'].append(examples['original_id'][index])
                    transf_data['prompt'].append(current_prompt)
                    transf_data['control_attribute'].append(args.control_attribute)
                    transf_data['control_attribute_model_value'].append([sent_model_value, actual_topic_value])
                    transf_data['control_attribute_value'].append([sent_val, topic_val])
    return transf_data


def execute_experiment(args: argparse.Namespace) -> None:
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)

    if args.control_attribute == "multiple":
        data_df = dataset_hf.map(transform_dataset_to_prompt_pplm_multiple, batched=True, 
                                remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                                fn_kwargs={'args': args})
    else:
        data_df = dataset_hf.map(transform_dataset_to_prompt_pplm, batched=True, 
                                remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                                fn_kwargs={'args': args})
    print('Prompts Dataset length:', len(data_df['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_tokens}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)[args.control_attribute]
    hyperparameters['length'] = args.max_tokens

    model, tokenizer = init_pplm(hyperparameters, args.control_attribute)

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

        texts = generate_text(batch['prompt'], model, tokenizer, hyperparameters, args.control_attribute, batch['control_attribute_value'], batch['control_attribute_model_value'])

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
                        choices=["sentiment", "topic", "multiple"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
