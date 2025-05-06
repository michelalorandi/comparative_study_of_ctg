from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
import pandas as pd
import torch
from tqdm import tqdm
import json
import os
from transformers import pipeline
from src.models.dexperts.generation import dexperts
from src.models.dexperts.utils import load_jsonl, batchify, ensure_dir


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        yield {'text': generation, **response}


def collate(dataset: pd.DataFrame, generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    assert len(generations) % len(dataset) == 0
    n = len(generations) // len(dataset)
    print(f"Detected samples per prompt:", n)
    generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
    dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)


def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, model_type: str, 
         pos_model: str, neg_model: str, positive: bool, n: int, max_tokens: int, batch_size: int, resume: bool,
         alpha: float, p: float, filter_p: float):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / 'generations.jsonl'
    sentiment_file = output_dir / 'sentiment.jsonl'
    assert resume or not os.path.exists(generations_file)
    ensure_dir(output_dir)
    output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'

    # Setup model for generation
    generations_iter = dexperts(
        prompts=prompts,
        max_len=max_tokens,
        num_samples=n,
        batch_size=batch_size,
        model_name_or_path=model,
        expert_name_or_path=pos_model,
        antiexpert_name_or_path=neg_model,
        out_file=generations_file,
        filter_p=filter_p,
        p=p,
        alpha=alpha,
    )

    # read generations
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)
    assert len(generations) % len(prompts) == 0
    n = len(generations) // len(prompts)

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis')
    with open(sentiment_file, 'w') as fo:
        for i, p in tqdm(enumerate(prompts), total=len(prompts), desc='Scoring generations'):
            sentences_for_prompt = []
            for j in range(n):
                gen = generations[i*n + j]
                sentences_for_prompt.append(f'{p}{gen}')
            try:
                predictions_for_prompt = classifier(sentences_for_prompt)
            except IndexError: # sometimes the generation is too long?
                predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
            for res in predictions_for_prompt:
                fo.write(json.dumps(res) + '\n')

    torch.cuda.empty_cache()
    print('Finished generation and sentiment scoring!')

    if os.path.exists(sentiment_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(sentiment_file), output_file)


if __name__ == '__main__':
    main()
