import os
import json
import pandas as pd
from src.utility_data import save_data, process_roc_stories, get_header_columns_dataset, load_json


def create_processed_dataframe(data: list):
    return pd.DataFrame(data, columns=["original_id", "prompt"])


def process_sts_benchmark(dataset_path: str, data_config: dict):
    with open(dataset_path, 'r') as data_file:
        raw_data = data_file.readlines()
    data = []
    for index in range(len(raw_data)):
        content = raw_data[index].split("\t")
        if content[0] in data_config['STSbenchmark']['allowed_types']:
            data.append([index, content[5]])
    return create_processed_dataframe(data)


def process_pplm_prompts(dataset_path: str):
    raw_data = load_json(dataset_path)
    data = []
    for key in raw_data:
        for index in range(len(raw_data[key])):
            data.append([f"{key}_{index}", raw_data[key][index]])
    return create_processed_dataframe(data)


def process_owt_prompts(dataset_path: str):
    with open(dataset_path, 'r') as data_file:
        lines = data_file.readlines()
    data = []
    for line in lines:
        obj = json.loads(line)
        data.append([obj['md5_hash'], obj['prompt']['text']])
    return create_processed_dataframe(data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', metavar='path', required=True)
    parser.add_argument("--output_path", metavar='path', required=True)
    parser.add_argument("--data_config_path", metavar='path', default="./scripts/src/config/data.json")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ROCStories", "STSbenchmark", "PPLMprompts", "cloze",
                                 "OpenWebText"])
    parser.add_argument("--subset_filepath", type=str, required=False)
    args = parser.parse_args()

    data_config = load_json(args.data_config_path)

    header, roc_columns = get_header_columns_dataset(data_config, args.dataset)

    processed_data_df = None
    if "ROCStories" in args.dataset_path or "cloze" in args.dataset_path:
        processed_data_df = process_roc_stories(args.dataset_path, roc_columns, header)
    elif "STSbenchmark" in args.dataset_path:
        processed_data_df = process_sts_benchmark(args.dataset_path, data_config)
    elif "PPLMprompts" in args.dataset_path:
        processed_data_df = process_pplm_prompts(args.dataset_path)
    elif "OpenWebText" in args.dataset_path:
        processed_data_df = process_owt_prompts(args.dataset_path)
    else:
        print('Error: dataset not available.')

    if processed_data_df is not None:
        save_data(os.path.join(args.output_path), processed_data_df)
