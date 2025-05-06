"""
 Script to execute the experiments using LLMs (LLaMa2 and Falcon LM)
"""
# standard libraries import
import time

# my scripts import
from src.utility_data import load_json
from src.utility_prompts import build_prompt
from src.utility_generation import get_model_type, get_dataset_name, get_task, load_control_values

# models related import
import transformers
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_huggingface_token(tokens_path):
    # Load your API key from an environment variable or secret management service
    tokens = load_json(tokens_path)
    global huggingface_access_token
    huggingface_access_token = tokens['huggingface_access_token']


def load_huggingface_model(model: str, seed: int):
    if 'llama' in model or 'falcon-40' in model:
        transformers.set_seed(seed)
        hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", load_in_4bit=True,
                                                        use_auth_token=huggingface_access_token)
        hf_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True,
                                                     use_auth_token=huggingface_access_token)
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        return hf_model, hf_tokenizer
    return None, None


def inference_loaded_model_batch(batch: list, model_hyperparams: dict, hf_model, hf_tokenizer):
    model_inputs = hf_tokenizer(batch, return_tensors="pt", padding=True).to("cuda:0")
    output = hf_model.generate(**model_inputs, **model_hyperparams)
    text = hf_tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    return str(output), text


def request_huggingface(prompt: str, model: str, model_hyperparams: dict, seed: int):
    while True:
        try:
            inference = InferenceClient(model, token=huggingface_access_token)
            response = inference.text_generation(prompt, seed=seed, **model_hyperparams)
            return response, response.generated_text

        except Exception as exception:
            print(str(exception))
            print('Retrying in 20 seconds')
            time.sleep(20)


def request_huggingface_batch(batch: list, model: str, model_hyperparams: dict, seed: int):
    responses = []
    texts = []
    for prompt in batch:
        response, text = request_huggingface(prompt, model, model_hyperparams, seed)
        responses.append(response)
        texts.append(text)
    return responses, texts


def llms_send_request_batch(batch: list, params: dict, hf_model, hf_tokenizer):
    if 'falcon-180' in params['model']:
        return request_huggingface_batch(batch, params['model'], params['hyperparameters'],
                                         params['seed'])
    if 'llama' in params['model'] or 'falcon-40' in params['model']:
        return inference_loaded_model_batch(batch, params['hyperparameters'], hf_model,
                                            hf_tokenizer)
    return None, None


def transform_dataset_to_prompt(examples: dict, args):
    model_type = get_model_type(args.model)
    dataset_name = get_dataset_name(args.dataset_filepath)

    task = get_task(args.dataset_filepath)
    control_values = load_control_values(args.control_attribute)

    transf_data = {
        "original_id": [],
        "prompt": [],
        "control_attribute": [],
        "control_attribute_value": []
    }
    for index in range(len(examples['prompt'])):
        for value in control_values:
            current_prompt = build_prompt(task, model_type, args.prompt_type,
                                          examples['prompt'][index], args.control_attribute, value,
                                          dataset_name)
            transf_data['original_id'].append(examples['original_id'][index])
            transf_data['prompt'].append(current_prompt)
            transf_data['control_attribute'].append(args.control_attribute)
            transf_data['control_attribute_value'].append(value)
    return transf_data


def transform_dataset_to_prompt_multiple(examples: dict, args):
    model_type = get_model_type(args.model)
    dataset_name = get_dataset_name(args.dataset_filepath)

    task = get_task(args.dataset_filepath)
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
                current_prompt = build_prompt(task, model_type, args.prompt_type,
                                            examples['prompt'][index], args.control_attribute, [sent_val, topic_val],
                                            dataset_name)
                transf_data['original_id'].append(examples['original_id'][index])
                transf_data['prompt'].append(current_prompt)
                transf_data['control_attribute'].append(args.control_attribute)
                transf_data['control_attribute_value'].append([sent_val, topic_val])
    return transf_data
