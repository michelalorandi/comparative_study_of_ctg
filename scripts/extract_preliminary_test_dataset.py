import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utility_data import save_data, process_roc_stories, get_header_columns_dataset, load_json


def get_sts_benchmark_dev(dataset_path):
    with open(dataset_path, 'r') as data_file:
        raw_data = data_file.readlines()
    data = []
    for index in range(len(raw_data)):
        content = raw_data[index].split("\t")
        data.append([index, content[5], f"{content[0]}_{content[1]}"])
    return pd.DataFrame(data, columns=["original_id", "prompt", "label"])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', metavar='path', required=True)
    parser.add_argument("--output_path", metavar='path', required=True)
    parser.add_argument("--data_config_path", metavar='path', default="./scripts/src/config/data.json")
    parser.add_argument("--examples_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=567)
    parser.add_argument("--dataset", type=str, required=True, choices=["ROCStories", "STSbenchmark", "PPLMprompts", "cloze"])
    parser.add_argument("--stratify_label", type=str, default=None)
    args = parser.parse_args()

    config = load_json(args.data_config_path)
    header, dataset_columns = get_header_columns_dataset(config, args.dataset)

    processed_data_df = None
    if args.dataset == "ROCStories" or args.dataset == "cloze":
        print()
        processed_data_df = process_roc_stories(args.dataset_path, dataset_columns, header)
    elif args.dataset == "STSbenchmark":
        processed_data_df = get_sts_benchmark_dev(args.dataset_path)
    elif args.dataset == "PPLMprompts":
        print()
        # processed_data_df = process_pplm_prompts(args.dataset_path)
    else:
        print('Error: dataset not available.')

    if processed_data_df is not None:
        stratify = processed_data_df[args.stratify_label] if args.stratify_label is not None else None
        _, examples_set_df, = train_test_split(processed_data_df, test_size=args.examples_size, random_state=args.seed,
                                               stratify=stratify, shuffle=True)
        if args.stratify_label is not None:
            examples_set_df = examples_set_df.drop(columns=[args.stratify_label])

        save_data(os.path.join(args.output_path), examples_set_df)
