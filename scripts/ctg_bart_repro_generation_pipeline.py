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
from src.utility_generation import load_model_hyperparameters, get_task

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
                   max_length: int, masked_sentences) -> List[str]:
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

    # construct encoder_inputs and indicate labels for bart
    indicate_labels_list = []
    encoder_inputs_list = []
    decoder_inputs_list = None
    for masked_sentence in  masked_sentences:
        indicate_labels = [0]
        encoder_inputs = [tokenizer.bos_token_id]
        words = masked_sentence.split()
        for i, w in enumerate(words):
            ids = tokenizer.encode(' '+w, add_special_tokens=False)
            encoder_inputs +=ids
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


def create_result_object(original_id, control_attribute_value, text, control_attribute_model_value=None):
    return {
            'original_id': original_id,
            'prompt': None,
            'control_attribute_value': control_attribute_value,
            'control_attribute_model_value': control_attribute_model_value,
            'sentence': text,
            'end_time_current_prompt': get_current_date()
        }


def get_results_object_batch(original_ids: list, control_attribute_values: List[str], texts: List[str], 
                             control_attribute_model_values: List[str] = None):
    results_obj = []
    for index, original_id in enumerate(original_ids):
        res = create_result_object(original_id, control_attribute_values[index],
                                   texts[index], control_attribute_model_values[index] if control_attribute_model_values is not None else None)
        results_obj.append(res)
    return results_obj



def execute_experiment(args: argparse.Namespace) -> None:

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset.split('/')[-1].split('.')[0]}-" \
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
            "task": 'reproduction',
            "dataset_filepath": args.dataset,
            "prompt": args.prompt_type,
            "seed": args.seed,
            "control_attribute": args.control_attribute,
            "model": args.model
        },
        'model_hyperparameters': hyperparameters,
        'results': []
    }
    results_filename = os.path.join(res_dir, f'raw_{experiment_name}.json')

    for num_keywords in [str(i) for i in range(1,7)]:
    # generate sentences with lexical constraints
        input_file = f'./data/{args.dataset}/{num_keywords}keywords.txt'
        print(f'Generate sentences with lexical constraints for {input_file}.')
        masked_sentences = []
        ground_truths = []
        with open(input_file) as fr:
            for i, line in enumerate(fr):
                if i%3==0:
                    continue
                else:
                    line = line.strip().split('\t')[1]
                    if i%3==1:
                        masked_sentences.append(line) # lexical constraints
                    elif i%3==2:
                        ground_truths.append(line)

        for index in tqdm(range(0, len(masked_sentences), args.batch_size),
                        desc=f"Executing {experiment_name}"):
            batch = masked_sentences[index:index+args.batch_size]
            texts = generate_texts(model, tokenizer, hyperparameters, args.max_length, batch)

            results['results'] += get_results_object_batch([f"id_{num_keywords}keywords_{str(index+i)}" for i in range(len(batch))], 
                                                            batch, 
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
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=[None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
