import os
from typing import List

from src.utility_data import load_json


def get_attribute_value_repr(prompts_config: dict, attribute: str, attribute_value):
    if prompts_config['attribute_representation'].get(attribute) is not None:
        repr = prompts_config['attribute_representation'][attribute]['representation']
        if isinstance(attribute_value, list):
            sep = prompts_config['attribute_representation'][attribute]['separator']
            text_value = [repr.format(value=value) for value in attribute_value]
            return sep.join(text_value)
        else:
            return repr.format(value=attribute_value)
    return attribute_value


def get_instruction_text(prompts_config: dict, prompt_type: str, prompt_text: str, attribute_value: str or List[str],
                         attribute: str):
    value = get_attribute_value_repr(prompts_config, attribute, attribute_value)

    if attribute == "multiple":
        return prompts_config['prompts'][prompt_type]['instruction_multiple'].format(prompt_text=prompt_text,
                                                                            sentiment_value=value[0],
                                                                            topic_value=value[1])
    return prompts_config['prompts'][prompt_type]['instruction'].format(prompt_text=prompt_text,
                                                                        attribute_value=value,
                                                                        attribute=attribute)


def get_system_description(prompts_config: dict, prompt_type: str, task: str):
    return prompts_config['prompts'][prompt_type]['system_descr'].format(
        task_name=prompts_config['tasks'][task]['task_name'],
        role=prompts_config['tasks'][task]['role'])


def get_system_text(prompts_config: dict, model_type: str, prompt_type: str, task: str):
    system_descr = get_system_description(prompts_config, prompt_type, task)
    return prompts_config['models'][model_type]['system_repr'].format(text=system_descr)


def get_user_text(prompts_config: dict, model_type: str, prompt_type: str, text: str,
                  attribute_value: str or List[str], attribute: str):
    instruction = get_instruction_text(prompts_config, prompt_type, text, attribute_value,
                                       attribute)
    return prompts_config['models'][model_type]['user_repr'].format(text=instruction)


def get_model_text(prompts_config: dict, model_type: str, text: str):
    return prompts_config['models'][model_type]['model_repr'].format(text=text)


def get_single_example_prompt(prompts_config: dict, prompt_type: str, model_type: str,
                              input_text: str, output_text: str, control_value: str or List[str],
                              control_attribute: str):
    user_text = get_user_text(prompts_config, model_type, prompt_type, input_text, control_value,
                              control_attribute)
    model_text = get_model_text(prompts_config, model_type, output_text)
    return user_text + model_text


def get_examples_prompt(prompts_config: dict, prompt_type: str, task: str, model_type: str,
                        control_attribute: str):
    if prompts_config['prompts'][prompt_type].get("examples") is not None:
        examples = []
        for example in prompts_config['prompts'][prompt_type]['examples'][task]:
            example_attributes = example[control_attribute] if control_attribute != "multiple" else [example["sentiment"], example["topic"]]
            example_text = get_single_example_prompt(prompts_config, prompt_type, model_type,
                                                     example['input'],
                                                     example['output'], example_attributes,
                                                     control_attribute)
            examples.append(example_text)
        return ''.join(examples)
    return ''


def build_prompt(task: str, model_type: str, prompt_type: str, sample: str, control_attribute: str,
                 control_value: str or List[str], dataset: str):
    prompts = load_json(os.path.join('.', 'scripts', 'src', 'config', 'prompts.json'))

    system_text = get_system_text(prompts, model_type, prompt_type, task)
    user_text = get_user_text(prompts, model_type, prompt_type, sample, control_value,
                              control_attribute)

    examples_text = get_examples_prompt(prompts, prompt_type, task, model_type,
                                        control_attribute)
    
    start_sent = "Falcon: " if model_type == 'falcon' and prompt_type == "zero_shot" else ""

    return system_text + examples_text + user_text + start_sent
