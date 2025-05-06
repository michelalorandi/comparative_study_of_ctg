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
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.models.ctg_bart.models.main import generate
from src.models.ctg_bart.language_models.language_model import LanguageModel
from src.models.ctg_bart.src.transformers import BartForTextInfill, BartTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(hyperparameters: dict) -> (BartForTextInfill, BartTokenizer):
    tokenizer = BartTokenizer.from_pretrained(hyperparameters['model_path'])
    model = BartForTextInfill.from_pretrained(hyperparameters['model_path'])
    model = model.to(device)

    return model, tokenizer


def generate_texts(model: BartForTextInfill, tokenizer: BartTokenizer, hyperparameters: dict, 
                   max_length: int, prompts: List[str], keywords: List[str]) -> List[str]:
    texts = []

    stop_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)
    sub_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)

    filename = './scripts/src/models/ctg_bart/tokens/bart_stop_tokens.txt'
    index = 0
    with open(filename, 'r') as fr:
        for line in fr:
            words = line.strip().split()
            token_id = int(words[0])
            stop_tokens_tensor[token_id] = 1
            index += 1

    # load sub tokens
    filename = './scripts/src/models/ctg_bart/tokens/bart_sub_tokens.txt'
    index = 0
    with open(filename, 'r') as fr:
        for line in fr:
            words = line.strip().split()
            token_id = int(words[0])
            sub_tokens_tensor[token_id] = 1
            index += 1

    if hyperparameters['decoder_chain']>1:
        try:
            rank_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            rank_model = GPT2LMHeadModel.from_pretrained('gpt2')
        except:
            raise ValueError('can not load models.')
        rank_lm =  LanguageModel(device, rank_model, rank_tokenizer)
    else:
        rank_lm = None

    # generate sentences with lexical constraints

    # construct encoder_inputs and indicate labels for bart
    indicate_labels_list = []
    encoder_inputs_list = []
    decoder_inputs_list = None
    for prompt in  prompts:
        indicate_labels = [0]
        encoder_inputs = [tokenizer.bos_token_id]
        # Encode the prompts and assign 0 to each token composing the prompt
        # 0 means that it's not possible to add new tokens before the current token
        # 1 means that it's possible to add new tokens before the current token
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        encoder_inputs += encoded_prompt
        indicate_labels += [0]*len(encoded_prompt)
        for i, w in enumerate(keywords):
            ids = tokenizer.encode(' '+w, add_special_tokens=False)
            encoder_inputs += ids
            indicate_labels+=[1]+[0]*(len(ids)-1) # can insert before the current token
        encoder_inputs.append(tokenizer.eos_token_id)
        indicate_labels.append(1)
        indicate_labels_list.append(indicate_labels)
        encoder_inputs_list.append(encoder_inputs)

    encoder_inputs_list = [torch.tensor(e) for e in encoder_inputs_list]
    if decoder_inputs_list is not None:
        decoder_inputs_list = [torch.tensor(e) for e in decoder_inputs_list]

    if decoder_inputs_list is not None:
        decoder_inputs = decoder_inputs_list
    else:
        decoder_inputs = None
    predict_outputs, refinement_steps = generate(model, tokenizer, encoder_inputs_list, indicate_labels_list,
            hyperparameters['encoder_loss_type'],
            hyperparameters['max_insert_label'],
            device,
            decoder_inputs = decoder_inputs,
            stop_tokens_tensor = stop_tokens_tensor,
            sub_tokens_tensor =  sub_tokens_tensor,
            temperature=hyperparameters['temperature'],
            do_sample=hyperparameters['do_sample'],
            top_k=hyperparameters['top_k'],
            top_p=hyperparameters['top_p'],
            refinement_steps=hyperparameters['refinement_steps'],
            max_refinement_steps=['max_refinement_steps'],
            adaptive=hyperparameters['adaptive'],
            repetition_penalty=hyperparameters['repetition_penalty'],
            threshold=hyperparameters['threshold'],
            decoder_chain=hyperparameters['decoder_chain'],
            rank_lm=rank_lm,
            max_len = max_length

    )
    for b in range(len(predict_outputs)):
        texts.append(tokenizer.decode(predict_outputs[b].tolist()[1:-1], clean_up_tokenization_spaces=False))
    

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

    model, tokenizer = init_model(hyperparameters)

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
            texts = generate_texts(model, tokenizer, hyperparameters, args.max_length, batch['prompt'], control_val)

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
