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

from transformers import AutoModelForSequenceClassification, GPT2TokenizerFast, AdamW


SENTIMENT = {
    "positive": "pos",
    "negative": "neg"
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(hyperparameters: dict, control_attribute: str, control_attribute_value: str) -> (any, GPT2TokenizerFast):
    if control_attribute == "sentiment":
        from src.models.bolt.model_with_biases import GPTPromptTuningWithbiasesModelLM
    else:
        from src.models.bolt.keywords_model_with_biases import GPTPromptTuningWithbiasesModelLM
    
    tokenizer = GPT2TokenizerFast.from_pretrained(hyperparameters['tokenizer'])
    tokenizer.pad_token = tokenizer.eos_token

    model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
        hyperparameters['model'],
        n_tokens=hyperparameters['n_prompt_tokens'],
        initialize_from_vocab=hyperparameters['init_from_vocab'],
        use_full_prompt=False,
    )
    model.cuda()

    if control_attribute == "sentiment":
        discriminator = AutoModelForSequenceClassification.from_pretrained(hyperparameters['discriminator'])
        discriminator.cuda()
        model.init_discriminator(discriminator)

    return model, tokenizer


def generate_texts_sentiment(model: any, 
                             tokenizer: GPT2TokenizerFast, 
                             batch_size: int, 
                             seq_len: int, 
                             prompts: List[str], 
                             control_attribute_value: str, 
                             hyperparameters: dict) -> List[str]:
    texts = []

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], SENTIMENT[control_attribute_value])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "biases" in n or "trainable_weights" in n],
            "weight_decay": hyperparameters['weight_decay'],
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparameters['learning_rate'])
    model.eval()
    
    for i in range(8):
        if i % 1 == 0:
            loss, output_ids, gpt_logit, senti_losses = model.soft_forward(**inputs, labels=inputs.input_ids, use_full_prompt=False)
            #print("Decoding: ", loss)
            sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            #print(sentences)
            # TODO check if this is correct
            texts = sentences

        loss.backward()
        if i % 1 == 0:
            optimizer.step()
            noise = [torch.normal(mean=0.01, std=0.01, size=model.biases[0].shape,
                                    device='cuda', requires_grad=False) for _ in range(len(model.biases))]
            for i in range(len(model.biases)):
                model.biases[i].data = model.biases[i].data + noise[i]

    return texts


def generate_texts_keywords(model: any, 
                            tokenizer: GPT2TokenizerFast, 
                            batch_size: int, 
                            seq_len: int, 
                            prompts: List[str], 
                            control_attribute_value: List[str], 
                            hyperparameters: dict) -> List[str]:
    texts = []

    keywords_word = [' '.join(control_attribute_value)] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    keywords = tokenizer([w for w in keywords_word], return_tensors="pt", padding=True)['input_ids']
    inputs = inputs.to("cuda")
    keywords = keywords.to("cuda")
    model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "biases" in n],
            "weight_decay": hyperparameters['weight_decay'],
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparameters['learning_rate'])
    model.eval()
    for i in range(100):
        #print("#################")
        loss, output_ids = model.soft_forward(**inputs, labels=inputs.input_ids, use_full_prompt=False, keywords=keywords)
        #print(keywords_word)
        sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #print(sentences)
        # TODO check if this is correct
        texts = sentences

        loss.backward()
        if i % 1 == 0:
            optimizer.step()
            noise = [torch.normal(mean=0.01, std=0.01, size=model.biases[0].shape,
                                    device='cuda', requires_grad=False) for _ in range(len(model.biases))]
            for i in range(len(model.biases)):
                model.biases[i].data = model.biases[i].data + noise[i]

    return texts


def execute_experiment(args: argparse.Namespace) -> None:
    control_values = load_control_values(args.control_attribute)

    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    print('Prompts Dataset length:', len(dataset_hf['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_length}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)[args.control_attribute]

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

        model, tokenizer = init_model(hyperparameters, args.control_attribute, control_val)

        for index in tqdm(range(0, len(dataset_hf['train']), args.batch_size),
                        desc=f"Executing {experiment_name} {control_val}"):
            
            batch = dataset_hf['train'][index:index+args.batch_size]
            if args.control_attribute == "keywords":
                texts = generate_texts_keywords(model, tokenizer, args.batch_size, args.max_length, batch['prompt'], control_val, hyperparameters)
            else:
                texts = generate_texts_sentiment(model, tokenizer, args.batch_size, args.max_length, batch['prompt'], control_val, hyperparameters)

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
                        choices=["sentiment", "keywords"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
