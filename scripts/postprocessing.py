import os
import argparse

import pandas as pd

from src.utility_data import load_json, save_data, extract_falcon_response, extract_llama_response, \
    extract_gedi_response, extract_pplm_response, extract_ctrl_response, extract_cat_paw_response, \
    extract_falcon40_response


def process_experiment(raw_res_filepath, control_attribute):
    raw_results = load_json(raw_res_filepath)

    processed_results = []
    for res in raw_results["results"]:
        text = res["sentence"]

        if 'falcon-40' in raw_res_filepath:
            processed_text = extract_falcon40_response(text, res['prompt'])
        elif 'Llama' in raw_res_filepath:
            processed_text = extract_llama_response(text, res['prompt'])
        elif 'gedi' in raw_res_filepath:
            processed_text = extract_gedi_response(text)
        elif 'pplm' in raw_res_filepath.split('-')[0]:
            processed_text = extract_pplm_response(text, res['prompt'])
        elif 'cat_paw' in raw_res_filepath:
            processed_text = extract_cat_paw_response(text)
        elif 'ctrl' in raw_res_filepath:
            processed_text = extract_ctrl_response(text)
        else:
            processed_text = text

        if control_attribute == 'keywords' or control_attribute == 'multiple':
            control_attribute_value = '#'.join(res['control_attribute_value'])
        else:
            control_attribute_value = res['control_attribute_value']

        processed_results.append({
            "id": f"{res['original_id']}_{res['control_attribute_value']}",
            'original_id': res['original_id'],
            "prompt": res["prompt"],
            "control_attribute_value": control_attribute_value,
            "text": processed_text
        })
    return processed_results


def process_all_experiments(results_root: str, control_attribute: str, model: str):
    control_attr_path = os.path.join(results_root, control_attribute)
    for folder in os.listdir(control_attr_path):
        if folder.startswith(model):
            folder_path = os.path.join(control_attr_path, folder)

            raw_res_filepath = os.path.join(folder_path, f'raw_{folder}.json')
            processed_res_filepath = os.path.join(folder_path, f'processed_{folder}.csv')

            #if os.path.exists(processed_res_filepath):
            if False:
                print(f"Skipping {folder} because the processed file already exists\n")
                continue
            if not os.path.exists(raw_res_filepath):
                print(f"Skipping {folder} because the raw file does not exist\n")
                continue
            print(f"Processing {folder}\n")
            processed_results = process_experiment(raw_res_filepath, control_attribute)

            res_df = pd.DataFrame.from_records(processed_results)
            save_data(processed_res_filepath, res_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results'))
    parser.add_argument('--control_attribute', type=str, required=True,
                        choices=["sentiment", "topic", "keywords", "multiple"])
    parser.add_argument('--model', type=str, required=True)
    arguments = parser.parse_args()

    process_all_experiments(arguments.results_folder_path, arguments.control_attribute, arguments.model)
