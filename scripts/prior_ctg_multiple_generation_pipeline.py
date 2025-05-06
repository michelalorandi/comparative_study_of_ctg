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
import math
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from functools import partial

# my scripts import
from src.utility_data import save_json
from src.utility_general import get_current_date, check_folder_exists_and_create
from src.utility_generation import load_model_hyperparameters, get_task, transform_dataset_to_prompt_general, get_results_object_batch, load_control_values

# models related import
import torch
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer

from src.models.prior_control.model import AE
from src.models.prior_control.latentops_modules import DIS, DIScons, sample_q_ode


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
        self.is_constrained = hyperparameters['is_constrained']

        self.weight = json.dumps(hyperparameters['weight'])
        self.optim_weight = json.dumps(hyperparameters['optim_weight'])
        self.config = hyperparameters['config']
        self.optim_config = hyperparameters['optim_config']


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


def calculate_multiple(alpha, mean, std, eps):
    total = sum(alpha)
    alpha = [num/total for num in alpha]
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

def gaussian_log_prob(x, mu, log_sd):
    return -0.5 * math.log(2 * torch.pi) - log_sd - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sd)


def prior_control_generate_multiple(batch: List[str], control_attribute_values: List[str], model: AE, 
                                    decoder_tokenizer: GPT2Tokenizer, args: ModelParameters) -> List[str]:
    weight = json.loads(args.weight)
    std = args.std

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.loads(f.read())
            for keys in config:
                if keys == 'weight':
                    weight = config['weight']
                
                if keys == 'std':
                    std = config['std']

    if args.optim_config is not None:
        with open(args.optim_config, 'r') as f:
            optim_config = json.loads(f.read())
            for keys in config:
                if keys == 'weight':
                    optim_weight = optim_config['weight']

    if isinstance(weight, dict):
        default_weight = weight['default']
        weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
        for keys in weight:
            if keys != 'default':
                tmp_i = int(keys[0])
                tmp_j = int(keys[1])
                weight_dict[tmp_i][tmp_j] = weight[keys]
    else:
        weight_dict = [[weight for jt in range(4)]for it in range(2)]

    if isinstance(optim_weight, dict):
        default_weight = optim_weight['default']
        optim_weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
        for keys in optim_weight:
            if keys != 'default':
                tmp_i = int(keys[0])
                tmp_j = int(keys[1])
                optim_weight_dict[tmp_i][tmp_j] = optim_weight[keys]
    else:
        optim_weight_dict = [[optim_weight for jt in range(4)]for it in range(2)]

    priors = model.priors

    ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': args.latent_size}
    sampler = partial(sample_q_ode, device=device, **ode_kwargs)

    model.set_mode('normal')

    output_text = []
    for z in range(len(batch)):
        prompt = batch[z]
        control_attribute_value = control_attribute_values[z]
        sent_val = control_attribute_value[0]
        topic_val = control_attribute_value[1]

        weight = weight_dict[multi_control_index[sent_val]][multi_control_index[topic_val]]
        optim_weight = optim_weight_dict[multi_control_index[sent_val]][multi_control_index[topic_val]]

        if isinstance(std, list):
            tmp_std = std[multi_control_index[sent_val]*4+multi_control_index[topic_val]]
        else:
            tmp_std = std

        if args.is_constrained:
            dismodel = DIScons([priors[multi_control_index[sent_val]], priors[2+multi_control_index[topic_val]], priors[7]], [optim_weight[0], optim_weight[1], optim_weight[2]]).to(device)
        else:
            dismodel = DIS([priors[multi_control_index[sent_val]], priors[2+multi_control_index[topic_val]], priors[7]], [optim_weight[0], optim_weight[1], optim_weight[2]]).to(device)

        probs_raw = None
        probs_dis = None

        tokens = decoder_tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        input_ids = input_ids.expand(args.batch_size, -1)
        attention_mask = attention_mask.expand(args.batch_size, -1)

        #latents = torch.normal(0,1, (args.batch_size, args.latent_num * args.latent_size))
        #latents = torch.zeros(args.batch_size, args.latent_num * args.latent_size)
        eps = torch.normal(0,tmp_std, (args.batch_size, args.latent_size)).to(device)

        prior_head_index = [multi_control_index[sent_val],multi_control_index[topic_val]+2,7]

        learnable_prior_mean = [priors[p_index][0] for p_index in prior_head_index]
        learnable_prior_std = [torch.exp(priors[p_index][1]) for p_index in prior_head_index]

        z_k = calculate_multiple(weight, learnable_prior_mean, learnable_prior_std, eps)

        if probs_raw is None:
            probs_raw = [gaussian_log_prob(z_k, priors[multi_control_index[sent_val]][0], priors[multi_control_index[sent_val]][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                        gaussian_log_prob(z_k, priors[multi_control_index[topic_val]+2][0], priors[multi_control_index[topic_val]+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                        gaussian_log_prob(z_k, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size
                    ]
        else:
            probs_raw = [torch.concat([probs_raw[0], gaussian_log_prob(z_k, priors[multi_control_index[sent_val]][0], priors[multi_control_index[sent_val]][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                        torch.concat([probs_raw[1], gaussian_log_prob(z_k, priors[multi_control_index[topic_val]+2][0], priors[multi_control_index[topic_val]+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                        torch.concat([probs_raw[2], gaussian_log_prob(z_k, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0)
                    ]

        y = torch.tensor([prior_head_index] * args.batch_size).to(device)
        latent = sampler(ccf=dismodel, y=y, z_k=z_k.clone())

        if probs_dis is None:
            probs_dis = [gaussian_log_prob(latent, priors[multi_control_index[sent_val]][0], priors[multi_control_index[sent_val]][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                        gaussian_log_prob(latent, priors[multi_control_index[topic_val]+2][0], priors[multi_control_index[topic_val]+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                        gaussian_log_prob(latent, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size
                    ]
        else:
            probs_dis = [torch.concat([probs_dis[0], gaussian_log_prob(latent, priors[multi_control_index[sent_val]][0], priors[multi_control_index[sent_val]][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                        torch.concat([probs_dis[1], gaussian_log_prob(latent, priors[multi_control_index[topic_val]+2][0], priors[multi_control_index[topic_val]+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                        torch.concat([probs_dis[2], gaussian_log_prob(latent, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0)
                    ]            
        
        input_latent, _ = model.inv_flow(latent, rev=True)
        output = model.generate(
            input_latent=input_latent,
            input_ids=input_ids,
            attention_mask=attention_mask,
            variation=args.variation,
            max_len=args.max_len,
            rp=1.2
        )

        output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
    return output_text


def transform_dataset_to_prompt_multiple(examples: dict, args: argparse.Namespace) -> dict:
    sent_control_vals = load_control_values("sentiment")
    topic_control_vals = load_control_values("topic")

    transf_data = {
        "original_id": [],
        "prompt": [],
        "control_attribute": [],
        "control_attribute_value": []
    }
    for index in range(len(examples['prompt'])):
        for sent_val in sent_control_vals:
            for topic_val in topic_control_vals:
                current_prompt = examples['prompt'][index]
                transf_data['original_id'].append(examples['original_id'][index])
                transf_data['prompt'].append(current_prompt)
                transf_data['control_attribute'].append(args.control_attribute)
                transf_data['control_attribute_value'].append([sent_val, topic_val])
            
    return transf_data


def execute_experiment(args: argparse.Namespace) -> None:
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath, download_mode="force_redownload")

    data_df = dataset_hf.map(transform_dataset_to_prompt_multiple, batched=True, 
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

        texts = prior_control_generate_multiple(batch['prompt'], batch['control_attribute_value'], model, tokenizer, model_args)

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
                        choices=["multiple"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
