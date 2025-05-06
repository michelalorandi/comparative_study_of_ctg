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
from datasets import load_dataset, Dataset

# my scripts import
from src.utility_data import save_json
from src.utility_general import get_current_date, check_folder_exists_and_create
from src.utility_generation import load_model_hyperparameters, get_task, transform_dataset_to_prompt_general, get_results_object_batch, load_control_values

# models related import
import torch
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer

from src.models.multi_ctg.model import AE
from src.models.multi_ctg.generation_utils import KCenters


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
    def __init__(self, hyperparameters: dict, max_len: int, batch_size: int) -> None:
        self.pretrained_encoder = hyperparameters['pretrained_encoder']
        self.pretrained_decoder = hyperparameters['pretrained_decoder']
        self.latent_size = hyperparameters['latent_size']
        self.latent_num = hyperparameters['latent_num']
        self.seq_len_per_latent = hyperparameters['seq_len_per_latent']
        self.model_path = hyperparameters['model_path']
        self.batch_size = batch_size
        self.max_len = max_len
        self.seed = None
        self.variation = hyperparameters['variation']
        self.num_centers = hyperparameters['num_centers']
        self.num_output_centers = hyperparameters['num_output_centers']
        self.topk = hyperparameters['topk']
        self.batch = hyperparameters['batch']
        self.max_iter = hyperparameters['max_iter']
        self.strategy = hyperparameters['strategy']
        self.temperature = hyperparameters['temperature']
        self.SDM_reinit = hyperparameters['SDM_reinit']
        self.weight = hyperparameters['weight']
        self.rp = hyperparameters['rp']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_model(seed: int, hyperparameters: dict, max_len: int, batch_size: int) -> (AE, BertTokenizer, GPT2Tokenizer, ModelParameters):
    hyperparameters['max_len'] = max_len
    hyperparameters['batch_size'] = batch_size
    hyperparameters['seed'] = seed

    args = ModelParameters(hyperparameters, max_len, batch_size)
    args.seed = seed

    encoder_tokenizer = BertTokenizer.from_pretrained(hyperparameters['pretrained_encoder'])
    encoder = BertModel.from_pretrained(hyperparameters['pretrained_encoder'])
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(hyperparameters['pretrained_decoder'])
    decoder = GPT2LMHeadModel.from_pretrained(hyperparameters['pretrained_decoder'])
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    model = AE(encoder=encoder, decoder=decoder, args=args)
    model.load_state_dict(torch.load(hyperparameters['model_path']), strict=False)
    model.eval()

    model.to(device)
    return model, encoder_tokenizer, decoder_tokenizer, args


def get_dataloader(batch: List[str], encoder_tokenizer):
    dataset = Dataset.from_dict({'sent': batch})
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent']), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    return torch.utils.data.DataLoader(tmp_dataset, batch_size=32)


def get_latents(encoder_tokenizer: BertTokenizer, model: AE, num_centers: int, 
                latent_size: int, num_output_centers: List[int], max_length: int=500) -> (KCenters, dict, dict):
    imdb_dataset = [{'sent':[]} for i in range(2)]
    ag_dataset = [{'sent':[]} for i in range(4)]

    with open('./scripts/src/models/multi_ctg/data/IMDb/IMDb.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label = int(line[0])
            imdb_dataset[label]['sent'].append(line[1].strip())

    with open('./scripts/src/models/multi_ctg/data/AGnews/AG-data.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label = int(line[0])
            ag_dataset[label]['sent'].append(line[1].strip())
            label = int(line[0])
            ag_dataset[label]['sent'].append(line[1].strip())

    imdb_dataset = [Dataset.from_dict(i) for i in imdb_dataset]
    ag_dataset = [Dataset.from_dict(i) for i in ag_dataset]

    imdb_dataloader = []
    for dataset in imdb_dataset:
        tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=max_length, padding='max_length', truncation=True), batched=True)
        tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
        imdb_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

    ag_dataloader = []
    for dataset in ag_dataset:
        tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=max_length, padding='max_length', truncation=True), batched=True)
        tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
        ag_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

    sentiment_latents = {0:None, 1:None}
    topic_latents = {0:None, 1:None, 2:None, 3:None}

    for i in range(2):
        for cnt in tqdm(iter(imdb_dataloader[i])):
            encoder_input_ids = cnt['input_ids']
            encoder_attention_mask = cnt['attention_mask']
            encoder_token_type_ids = cnt['token_type_ids']
            
            latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
            if sentiment_latents[i] is None:
                sentiment_latents[i] = latent.squeeze().detach()
            else:
                sentiment_latents[i] = torch.cat((sentiment_latents[i], latent.squeeze().detach()), dim=0)

    for i in range(4):
        for cnt in tqdm(iter(ag_dataloader[i])):
            encoder_input_ids = cnt['input_ids']
            encoder_attention_mask = cnt['attention_mask']
            encoder_token_type_ids = cnt['token_type_ids']
            
            latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
            if topic_latents[i] is None:
                topic_latents[i] = latent.squeeze().detach()
            else:
                topic_latents[i] = torch.cat((topic_latents[i], latent.squeeze().detach()), dim=0)
    kcmodel = KCenters(num_centers=num_centers, latent_size=latent_size, num_output_centers=num_output_centers, device='cuda')
    return kcmodel, sentiment_latents, topic_latents


def generate_multiple(batch: List[str], control_attribute_values: List[str], model: AE, 
                      decoder_tokenizer: GPT2Tokenizer, args: ModelParameters, 
                      kcmodel: KCenters, sentiment_latents: dict, topic_latents: dict) -> List[str]:
    weight = args.weight
    num_output_centers = args.num_output_centers

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


    if isinstance(num_output_centers, int):
        num_output_centers = [[num_output_centers]*4]*2

    output_text = []
    for z in range(len(batch)):
        prompt = batch[z]
        control_attribute_value = control_attribute_values[z]
        sent_val = control_attribute_value[0]
        topic_val = control_attribute_value[1]

        i = multi_control_index[sent_val]
        j = multi_control_index[topic_val]

        weight = weight_dict[i][j]
        num_output_cent = num_output_centers[i][j]
        
        centers = kcmodel.train(
            [sentiment_latents[i].to('cuda'), topic_latents[j].to('cuda')],
            weight=weight,
            topk=args.topk,
            SDM_reinit=args.SDM_reinit,
            max_iter=args.max_iter,
            strategy=args.strategy,
            temperature=args.temperature,
            num_output_centers=num_output_cent
            ).cpu().numpy()
        centers = [torch.FloatTensor(k).unsqueeze(0) for k in centers]

        tokens = decoder_tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        input_ids = input_ids.expand(args.batch_size, -1)
        attention_mask = attention_mask.expand(args.batch_size, -1)

        output = model.generate(
            input_latent=random.choice(centers),
            input_ids=input_ids,
            attention_mask=attention_mask,
            variation=args.variation,
            max_len=args.max_len,
            rp=args.rp
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

    model, encoder_tokenizer, decoder_tokenizer, model_args = init_model(args.seed, hyperparameters, args.max_length, args.batch_size)

    kcmodel, sentiment_latents, topic_latents = get_latents(encoder_tokenizer, model, model_args.num_centers, model_args.latent_size, model_args.num_output_centers, args.max_length)

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

        texts = generate_multiple(batch['prompt'], batch['control_attribute_value'], model, decoder_tokenizer, 
                                  model_args, kcmodel, sentiment_latents, topic_latents)

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
