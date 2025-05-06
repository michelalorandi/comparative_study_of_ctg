import os

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.utility_data import save_json
from src.utility_general import check_folder_exists_and_create, get_current_date
from src.utility_generation import load_model_hyperparameters, get_task, transform_dataset_to_prompt_general, get_results_object_batch

import torch
from transformers import GPT2Config, GPT2Tokenizer

from src.models.gedi.modeling_gpt2 import GPT2LMHeadModel


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_gpu = torch.cuda.device_count()


topics = {
    "World": "world", 
    "Sports": "sports", 
    "Business": "business", 
    "Science/Technology": "science"
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_gedi(params, attribute):

    config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]

    gedi_model_name_or_path = params['gedi_model_name_or_path'][attribute]
    gen_model_name_or_path = params['gen_model_name_or_path']

    tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path,do_lower_case=False)
    model = model_class.from_pretrained(gen_model_name_or_path)
    model.to(device)

    gedi_model = model_class.from_pretrained(gedi_model_name_or_path)
    gedi_model.to(device)

    return model, tokenizer, gedi_model


def get_control_code(control_attribute_value):
    if topics.get(control_attribute_value) is not None:
        return topics[control_attribute_value]
    return control_attribute_value


def get_control_code_details(control_attribute, control_attribute_value, tokenizer):
    if control_attribute == "topic":
        multi_code = tokenizer.encode(control_attribute_value)
        code_desired = "true"
        code_undesired = "false"
    else:
        multi_code = None
        code_desired = control_attribute_value
        code_undesired = "negative" if control_attribute_value == "positive" else "positive"
    attr_class = 1
    return code_desired, code_undesired, multi_code, attr_class


def generate_text(batch, model, tokenizer, gedi_model, params, control_attribute, control_attribute_values):
    generated_texts = []

    for i in range(len(batch)):
        prompt = batch[i]
        control_attribute_value = control_attribute_values[i]

        code_desired, code_undesired, multi_code, attr_class = get_control_code_details(control_attribute, control_attribute_value, tokenizer)

        text_ids = tokenizer.encode(prompt)
        encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(device)

        generated_sequence = model.generate(
            input_ids=encoded_prompts,
            pad_lens=params['pad_lens'],
            max_length=params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            repetition_penalty=params['repetition_penalty'],
            rep_penalty_scale=params['rep_penalty_scale'],
            eos_token_ids=tokenizer.eos_token_id,
            pad_token_id=params['pad_token_id'],
            do_sample=params['do_sample'],
            penalize_cond=params['penalize_cond'],
            gedi_model=gedi_model,
            gpt3_api_key = params['gpt3_api_key'],
            tokenizer=tokenizer,
            disc_weight=params['disc_weight'],
            filter_p=params['filter_p'],
            target_p=params['target_p'],
            class_bias= params['class_bias'],
            attr_class=attr_class,
            code_0=code_undesired,
            code_1=code_desired,
            multi_code=multi_code
        )

        text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True)
        generated_texts.append(text)
    return generated_texts


def execute_experiment(args):
    dataset_hf = load_dataset("csv", data_files=args.dataset_filepath)
    data_df = dataset_hf.map(transform_dataset_to_prompt_general, batched=True, 
                              remove_columns=['original_id', 'prompt'], batch_size=args.batch_size,
                              fn_kwargs={'args': args})
    print('Prompts Dataset length:', len(data_df['train']))

    experiment_name = f"{args.model.split('/')[-1]}-" \
                      f"{args.dataset_filepath.split('/')[-1].split('.')[0]}-" \
                      f"{args.control_attribute}-{args.prompt_type}-len{args.max_tokens}-{args.seed}"

    hyperparameters = load_model_hyperparameters(args.model)
    hyperparameters['max_length'] = args.max_tokens

    model, tokenizer, gedi_model = init_gedi(hyperparameters, args.control_attribute)

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

        params = {
            'hyperparameters': hyperparameters,
            'model': args.model,
            'seed': args.seed
        }
        texts = generate_text(batch['prompt'], model, tokenizer, gedi_model, params['hyperparameters'], args.control_attribute, batch['control_attribute_value'])

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_path', metavar='path',
                        default=os.path.join('.', 'scripts', 'src', 'tokens', 'api_tokens.json'))
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--control_attribute', type=str, required=True,
                        choices=["sentiment", "topic", "keywords"])
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=20)
    parser.add_argument('--prompt_type', type=str, required=False, default=None,
                        choices=["zero_shot", "few_shot", None])
    arguments = parser.parse_args()

    set_seed(arguments.seed)

    execute_experiment(arguments)
