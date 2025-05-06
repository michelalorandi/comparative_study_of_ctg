import json
import os
import json
import pandas as pd


def load_json(filepath, option: str = "r"):
    with open(filepath, option) as file:
        return json.load(file)


def save_json(results, filepath: str):
    """

    Args:
        results:
        filepath:

    Returns:

    """
    with open(filepath, 'w') as res_file:
        json.dump(results, res_file)


def get_header_columns_dataset(config: dict, dataset_name: str):
    return config['header'], config[dataset_name]['column_names']


def save_data(output_filepath: str, data_df: pd.DataFrame):
    data_df.to_csv(output_filepath, sep=',', index=False)


def extract_falcon_response(text: str):
    response = []
    for line in text.split('\n'):
        if line.startswith('Falcon:'):
            line = line.replace('Falcon:', '').strip()
        elif line.startswith('User:'):
            return '\n'.join(response)
        response.append(line)
    return '\n'.join(response)


def extract_llama_response(text: str, prompt: str):
    prompt = prompt.replace(' </s><s>[INST]', '  [INST]').replace('<s>[INST] <<SYS>>',
                                                                  '[INST] <<SYS>>')
    return text.replace(prompt, '').strip()


def extract_falcon40_response(text: str, prompt: str):
    #prompt = prompt.replace(' </s><s>[INST]', '  [INST]').replace('<s>[INST] <<SYS>>', '[INST] <<SYS>>')
    text = text.replace(prompt, '').strip()
    response = []
    for line in text.split('\n'):
        if line.startswith('Falcon:'):
            line = line.replace('Falcon:', '').strip()
        elif line.startswith('User:'):
            return '\n'.join(response)
        response.append(line)
    return '\n'.join(response)


def extract_gedi_response(text: str):
    text = text.split('<|endoftext|>')[0]
    return text.strip()


def extract_pplm_response(text: str, prompt: str):
    prompt_wo_label = prompt.split(':')[0] + ':'
    final_text = text.split(prompt_wo_label)[1].split('<|endoftext|>')[0]
    return final_text.strip()


def extract_cat_paw_response(text: str):
    text = text.split('<|endoftext|>')[1]
    return text.strip()


def extract_ctrl_response(text: str):
    text_wo_label = text.split(' ')[1:]
    final_text = " ".join(text_wo_label)
    return final_text.strip()


def process_roc_stories(dataset_path: str, columns: list, header: list):
    data_df = pd.read_csv(dataset_path, sep=',')
    columns_dict = {}
    for index in range(len(header)):
        columns_dict[columns[index]] = header[index]
    processed_df = data_df[columns].rename(columns=columns_dict)
    return processed_df
