"""
 Script to execute the generation pipeline
"""
# standard libraries import
import os
import argparse
from typing import List
from datetime import datetime

# non-standard libraries import
import json
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# my scripts import
from src.utility_data import save_json
from src.utility_general import get_current_date, check_folder_exists_and_create
from src.utility_generation import load_model_hyperparameters, get_task, transform_dataset_to_prompt_general, get_results_object_batch, load_control_values

# models related import
import torch
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer

from src.models.prior_control.model import AE


l = {
    "negative": [0,-1,-1],
    "positive": [1, -1,-1],
    "World": [-1,0,-1],
    "Sports": [-1,1,-1],
    "Business": [-1,2,-1],
    "Science/Technology": [-1,3,-1]
}
alpha_attribute = {
    "negative":[1,0], 
    "positive": [-0.2, 1.2], 
    "World": [1.3,-0.1,-0.1,-0.1], 
    "Sports": [-0.1, 1.3, -0.1, -0.1], 
    "Business": [-0.1, -0.1, 1.3, -0.1], 
    "Science/Technology": [-0.1, -0.1, -0.1, 1.3]}
head_attribute = {
    "negative": [0, 1],
    "positive": [0, 1],
    "World": [2,3,4,5],
    "Sports": [2,3,4,5],
    "Business": [2,3,4,5],
    "Science/Technology": [2,3,4,5]
}
prior_head_index = {
    "negative": 0,
    "positive": 1,
    "World": 2,
    "Sports": 3,
    "Business": 4,
    "Science/Technology": 5
}
multi_control_index = {
    "negative": 0,
    "positive": 1,
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Science/Technology": 3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelParameters:
    def __init__(self, hyperparameters) -> None:
        self.pretrained_encoder = hyperparameters['pretrained_encoder']
        self.pretrained_decoder = hyperparameters['pretrained_decoder']
        self.latent_size = hyperparameters['latent_size']
        self.latent_num = hyperparameters['latent_num']
        self.seq_len_per_latent = hyperparameters['seq_len_per_latent']
        self.model_path = hyperparameters['model_path']
        self.batch_size = hyperparameters['batch_size']
        self.max_len = hyperparameters['max_length']
        self.seed = None
        self.variation = hyperparameters['variation']
        self.rp = hyperparameters['rp']

        #Parameters for Prior
        self.prior = hyperparameters['prior']
        self.flow_num = hyperparameters['flow_num']
        self.prior_num = hyperparameters['prior_num']

        #Generation
        self.std = hyperparameters['std']

        self.is_extend = hyperparameters['is_extend']

        self.weight = json.dumps(hyperparameters['weight'])
        self.config = hyperparameters['config']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_prior_model(seed: int, hyperparameters: dict) -> (AE, GPT2Tokenizer, ModelParameters):
    args = ModelParameters(hyperparameters)

    # encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
    args.seed = seed

    encoder = BertModel.from_pretrained(args.pretrained_encoder)
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
    decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    model = AE(encoder=encoder, decoder=decoder, args=args)


    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()
    model.fix_decoder()
    model.set_mode('prior')

    model.to(device)
    return model, decoder_tokenizer, args


def calculate(alpha, mean, std, eps):
    mu = 0
    for w, mean in zip(alpha, mean):
        mu = mu + w * mean

    sigma = 0
    for w, std in zip(alpha, std):
        if w < 0:
            w = 0
        if w > 1:
            w = 1
        sigma = sigma + (w * std)**2
    sigma = torch.sqrt(sigma)
    
    sampled_dis = sigma * eps + mu
    return sampled_dis


def prior_control_generation(batch: List[str], control_attribute_values: List[str], model: AE, 
                             decoder_tokenizer: GPT2Tokenizer, args: ModelParameters) -> List[str]:
    output_text = []

    for i in range(len(batch)):
        prompt = batch[i]
        control_attribute_value = control_attribute_values[i]
        tokens = decoder_tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        input_ids = input_ids.expand(1, -1)
        attention_mask = attention_mask.expand(1, -1)

        #latents = torch.normal(0,1, (batch_size, latent_num * latent_size))
        latents = torch.zeros(1, args.latent_num * args.latent_size)

        if args.is_extend:
            output = model.generate(
                input_latent=latents,
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=args.variation,
                max_len=args.max_len,
                rp=args.rp,
                prior_head_index=head_attribute[control_attribute_value],
                alpha=alpha_attribute[control_attribute_value],
                calculate=calculate,
                std=args.std
            )
        else:
            output = model.generate(
                input_latent=latents,
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=args.variation,
                max_len=args.max_len,
                rp=args.rp,
                prior_head_index=prior_head_index[control_attribute_value],
                std=args.std
            )

        output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
    return output_text


def execute_experiment(args: argparse.Namespace) -> None:
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath, download_mode="force_redownload")

    data_df = dataset_hf.map(transform_dataset_to_prompt_general, batched=True, 
                            remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                            fn_kwargs={'args': args})
    print('Prompts Dataset length:', len(data_df['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['max_length'] = args.max_length

    model, tokenizer, model_args = init_prior_model(args.seed, hyperparameters)

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

        texts = prior_control_generation(batch['prompt'], batch['control_attribute_value'], model, tokenizer, model_args)

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

    parser = argparse.ArgumentParser()
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
