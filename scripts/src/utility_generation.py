import os
from typing import List

from src.utility_data import load_json
from src.utility_general import get_current_date

config_data = load_json(os.path.join('.', 'scripts', 'src', 'config', 'data.json'))
dataset_to_task = config_data['dataset_to_task']


def get_dataset_name(dataset_path: str):
    """

    Args:
        dataset_path:

    Returns:

    """
    for dataset in dataset_to_task:
        if dataset in dataset_path:
            return dataset
    return None


def get_task(dataset_filepath: str):
    """

    Args:
        dataset_filepath:

    Returns:

    """
    dataset_name = get_dataset_name(dataset_filepath)
    if dataset_name is None:
        return "original"
    return dataset_to_task[dataset_name]


def get_model_type(model: str):
    """

    Args:
        model:

    Returns:

    """
    if "falcon" in model:
        return "falcon"
    if "llama" in model:
        return "llama"
    return model


def load_model_hyperparameters(model: str):
    """

    Args:
        model:

    Returns:

    """
    model_type = get_model_type(model)
    hyperparams = load_json(os.path.join('.', 'scripts', 'src', 'config', 'hyperparameters.json'))
    return hyperparams[model_type]


def load_control_values(control_attribute: str or List[str]):
    """

    Args:
        control_attribute:

    Returns:

    """
    if type(control_attribute) == str:
        control_attribute = [control_attribute]
    control_config = load_json(os.path.join('.', 'scripts', 'src', 'config', 'control.json'))
    values = []
    for attribute in control_attribute:
        values += control_config[attribute]
    return values


def transform_dataset_to_prompt_general(examples: dict, args):
    control_values = load_control_values(args.control_attribute)

    transf_data = {
        "original_id": [],
        "prompt": [],
        "control_attribute": [],
        "control_attribute_value": []
    }
    for index in range(len(examples['prompt'])):
        for value in control_values:
            current_prompt = examples['prompt'][index]
            transf_data['original_id'].append(examples['original_id'][index])
            transf_data['prompt'].append(current_prompt)
            transf_data['control_attribute'].append(args.control_attribute)
            transf_data['control_attribute_value'].append(value)
    return transf_data


def create_result_object(original_id, prompt, control_attribute_value, text, control_attribute_model_value=None):
    return {
            'original_id': original_id,
            'prompt': prompt,
            'control_attribute_value': control_attribute_value,
            'control_attribute_model_value': control_attribute_model_value,
            'sentence': text,
            'end_time_current_prompt': get_current_date()
        }


def get_results_object_batch(original_ids: list, prompts: List[str],
                             control_attribute_values: List[str], texts: List[str], 
                             control_attribute_model_values: List[str] = None):
    results_obj = []
    for index, original_id in enumerate(original_ids):
        res = create_result_object(original_id, prompts[index], control_attribute_values[index],
                                   texts[index], control_attribute_model_values[index] if control_attribute_model_values else None)
        results_obj.append(res)
    return results_obj
