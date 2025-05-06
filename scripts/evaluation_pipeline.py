import os
import math
import argparse
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.distinct_n import eval_distinct
from src.CommonGenCoverage import coverageScore, originalCoverageScore
from src.utility_data import save_json, load_json
from src.utility_generation import load_control_values

import torch
from evaluate import load
from datasets import load_dataset, Dataset

from transformers import Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2LMHeadModel


control_attribute2id = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Science/Technology": 3,
    "positive": 1,
    "negative": 0
}
topics = ['World', 'Sports', 'Business', 'Science/Technology']


class Label():
    def __init__(self, label):
        self.label = label


lbl = Label(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_control_attribute_label(classifier_label, config: dict) -> str:
    if classifier_label in config['labels']:
        return config['labels'][classifier_label]
    return classifier_label


def load_perplexity_model() -> object:
    return load("perplexity", module_type="metric")


def compute_batch_perplexity(batch: List[str], model: str, perplexity, ground_truth: List[str], 
                             control_values: List[str]) -> dict:
    perplexity_results = {'overall': perplexity.compute(predictions=batch, model_id=model)}
    for label in control_values:
        lbl_batch = [batch[index] for index in range(len(batch)) if ground_truth[index] == label]
        if len(lbl_batch) > 0:
            perplexity_results[label] = perplexity.compute(predictions=lbl_batch, model_id=model)
    return perplexity_results


def load_hf_automodels(model: dict) -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForSeq2SeqLM.from_pretrained(model['model']).to("cuda:0")
    return tokenizer, model, None


def load_hf_autoclassifier(model: dict) -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForSequenceClassification.from_pretrained(model['model']).to("cuda:0")
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0)
    return tokenizer, model, classifier


def load_hf_distilbert_model(model: dict) -> Tuple[object, object]:
    tokenizer = DistilBertTokenizer.from_pretrained(model['tokenizer'])
    model = DistilBertForSequenceClassification.from_pretrained(model['model']).to("cuda:0")
    return tokenizer, model, None


def load_prior_model(model: dict):
    tokenizer = DebertaV2Tokenizer.from_pretrained(model['tokenizer'])
    model = DebertaV2ForSequenceClassification.from_pretrained(model['model'], num_labels=2)
    return tokenizer, model, None


def load_gpt2_automodels(model: dict) -> Tuple[object, object]:
    tokenizer = GPT2Tokenizer.from_pretrained(model['tokenizer'])
    model = GPT2LMHeadModel.from_pretrained(model['model']).to("cuda:0")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, None


def load_causal_lm(model: dict) -> Tuple[object, object, None]:
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForCausalLM.from_pretrained(model['model']).to('cuda:0')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, None
    


def load_tokenizer_and_model(model: dict) -> Tuple[object, object]:
    if model['model'] == 'distilbert-base-uncased-finetuned-sst-2-english':
        return load_hf_distilbert_model(model)
    elif 'Yelp2-checkpoint-64000' in model['model'] or 'AGnews-checkpoint-6000' in model['model']:
        return load_prior_model(model)
    elif model['model'] == 'textattack/distilbert-base-uncased-ag-news' or model['model'] == 'fabriceyhc/bert-base-uncased-ag_news':
        return load_hf_autoclassifier(model)
    elif 'gpt2' in model['model']:
        return load_gpt2_automodels(model)
    elif 'bloom' in model['model']:
        return load_causal_lm(model)
    return load_hf_automodels(model)


def compute_metrics_prior(pred):
    labels = torch.tensor(pred.label_ids).long()
    preds = torch.softmax(torch.tensor(pred.predictions),dim=-1)
    probs = torch.gather(preds, 1,labels.view(-1, 1))
    acc = torch.mean(probs).item()

    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=lbl.label)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'preds': preds.tolist(),
        'probs': np.squeeze(np.array(probs.tolist()), axis=1)
    }


def get_batch_prediction_ditilbert_class(batch: List[str], tokenizer, model, 
                                         config: dict) -> Tuple[List[str], List[list], None, None]:
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
    with torch.no_grad():
        logits = model(**inputs).logits

    preds = []
    for sample in logits:
        predicted_class_id = sample.argmax().item()
        pred_class = model.config.id2label[predicted_class_id]
        preds.append(get_control_attribute_label(pred_class, config))
    return preds, logits, None, None


def get_batch_pred_t5_class(batch: List[str], tokenizer, model, config) -> Tuple[List[str], None, None]:
    updated_batch = [f"sentiment: {text}" for text in batch]
    inputs = tokenizer(updated_batch, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda:0")
    model_preds = model.generate(inputs, max_new_tokens=5)
    decoded_preds = tokenizer.batch_decode(sequences=model_preds, skip_special_tokens=True)
    preds = []
    for pred_class in decoded_preds:
        preds.append(get_control_attribute_label(pred_class, config))
    return preds, None, None, None


def get_batch_pred_topic_pipeline(batch: List[str], classifier) -> Tuple[List[str], None, None]:
    class_names = ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"]
    id2label = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    outputs = classifier(batch, **tokenizer_kwargs)
    preds = [id2label[class_names.index(o['label'])] for o in outputs]
    return preds, None, None, None


def get_batch_pred_prior_classif(dataset: Dataset, control_attribute: str, tokenizer: DebertaV2Tokenizer, 
                                 model: DebertaV2ForSequenceClassification, batch_size: int, 
                                 label: str=None, length: int=100) -> Tuple[List[str], None, dict, None]:
    test_args = TrainingArguments(
        output_dir='logs',
        do_train = False,
        do_predict = True,
        no_cuda = False,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last = False,
        report_to='none'
    )

    length += 20

    wrap_preds = None
    if control_attribute == 'topic':
        all_probs = {}
        all_preds = {}
        for top in topics:
            eval_dataset = dataset.map(lambda e: tokenizer(top+'[SEP]'+e['sent'], truncation=True, padding='max_length', max_length=length))
            eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
            lbl.label = 1
            eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

            trainer = Trainer(
                model=model,
                args = test_args,
                compute_metrics=compute_metrics_prior, 
            )
            eval_res = trainer.evaluate(eval_dataset)
            all_probs[top] = eval_res['eval_probs']
            all_preds[top] = eval_res['eval_preds']
        final = np.array([all_probs[t] for t in topics]).T
        final_preds = np.array([all_preds[t] for t in topics]).T
        final_probs = final*final_preds
        wrap_preds = list(final_probs.argmax(-1))


    eval_dataset = None
    if control_attribute == 'topic':
        eval_dataset = dataset.map(lambda e: tokenizer(topics[e['label']]+'[SEP]'+e['sent'], truncation=True, padding='max_length', max_length=length))
        eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
        lbl.label = 1
    else:
        eval_dataset = dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=length), batched=True)
        eval_dataset = eval_dataset.map(lambda e: {'labels': e['label']})
        lbl.label = control_attribute2id[label] if label is not None else 1
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    trainer = Trainer(
        model=model,
        args = test_args,
        compute_metrics=compute_metrics_prior, 
    )

    eval_res = trainer.evaluate(eval_dataset)

    metrics_res = {
        'accuracy': eval_res['eval_accuracy'],
        'f1': eval_res['eval_f1'],
        'precision': eval_res['eval_precision'],
        'recall': eval_res['eval_recall']
    }

    return eval_res['eval_preds'], None, metrics_res, wrap_preds


def evaluate_results_prior(filepath: str, control_attribute: str, tokenizer: DebertaV2Tokenizer, 
                           model: DebertaV2ForSequenceClassification, batch_size: int, 
                           length: int=100) -> Tuple[None, None, dict, None]:
    eval_results = {}

    df = pd.read_csv(filepath.replace('evaluation_results_', 'processed_').replace('.json', '.csv'))

    if control_attribute.startswith('multiple'):
        if control_attribute.endswith('sentiment'):
            idx = 0
        else:
            idx = 1
        tmp_dataset = {
            'control_attribute_value': [att.split('#')[idx] for att in df['control_attribute_value'].tolist()],
            'text': df['text'].tolist()
        }
        df = pd.DataFrame.from_dict(tmp_dataset)
        control_attribute = 'sentiment' if 'sentiment' in control_attribute else 'topic'

    dataset = {
        'label': [control_attribute2id[att] for att in df['control_attribute_value'].tolist()], 
        'sent': [str(t) for t in df['text'].tolist()]
    }
    dataset = Dataset.from_dict(dataset)
    complete_preds, _, eval_results['overall'], wrapper_preds = get_batch_pred_prior_classif(dataset, control_attribute, tokenizer, model, batch_size, length=length)

    control_values = load_control_values(control_attribute)

    if control_attribute == 'sentiment':
        for label in control_values:
            dataset = {'label': [], 'sent': []}
            ctr_values = df['control_attribute_value'].tolist()
            texts = df['text'].tolist()
            for index in range(len(ctr_values)):
                if ctr_values[index] == label:
                    dataset['label'].append(control_attribute2id[label])
                    dataset['sent'].append(str(texts[index]))
            if len(dataset['label']) > 0:
                dataset = Dataset.from_dict(dataset)
                preds, _, eval_results[label], _ = get_batch_pred_prior_classif(dataset, control_attribute, tokenizer, model, batch_size, label=label, length=length)
    return complete_preds, None, eval_results, wrapper_preds


def get_batch_prediction_hf_class(batch: List[str], tokenizer, model, config: dict,
                                  model_name: str, filepath: str, control_attribute: str,
                                  batch_size: int, length: int, classifier) -> Tuple[List[str], List[list], dict, List[str]]:
    if model_name == 'distilbert-base-uncased-finetuned-sst-2-english':
        return get_batch_prediction_ditilbert_class(batch, tokenizer, model, config)
    elif model_name == 'michelecafagna26/t5-base-finetuned-sst2-sentiment':
        return get_batch_pred_t5_class(batch, tokenizer, model, config)
    elif 'Yelp2-checkpoint-64000' in model_name or 'AGnews-checkpoint-6000' in model_name:
        return evaluate_results_prior(filepath, control_attribute, tokenizer, model, batch_size, length)
    elif model_name == 'textattack/distilbert-base-uncased-ag-news' or model_name == 'fabriceyhc/bert-base-uncased-ag_news':
        return get_batch_pred_topic_pipeline(batch, classifier)
    return None, None, None, None


def compute_acc_prec_rec_f1(predictions: List[str], ground_truth: List[str], pos_label=None, average='binary') -> dict:
    # compute accuracy, precision, recall, f1
    results = {
        'accuracy': accuracy_score(ground_truth, predictions)*100,
        'f1': f1_score(ground_truth, predictions, pos_label=pos_label, average=average)*100,
        'precision': precision_score(ground_truth, predictions, pos_label=pos_label, average=average)*100,
        'recall': recall_score(ground_truth, predictions, pos_label=pos_label, average=average)*100
    }
    return results


def word_coverage_from_bolt(preds, keywords):
    soft_successes = 0
    strict_successes = 0
    for idx in range(len(preds)):
        matches = [keyword in preds[idx] for keyword in keywords[idx]]
        if any(matches):
            soft_successes += 1
        if all(matches):
            strict_successes += 1
    return soft_successes/len(preds), strict_successes/len(preds)


def compute_multiple_control_metrics(predictions_filepath: str, eval_res_path: str) -> None:
    eval_config = load_json(os.path.join('.', 'scripts', 'src', 'config', 'evaluation.json'))

    eval_results = load_json(eval_res_path)
    eval_preds = load_json(predictions_filepath)

    # load sentiment classifiers
    sent_classifiers = [c['model'].split('/')[-1] if 'Yelp2-checkpoint-64000' in c['model'] else c['model'] for c in eval_config['classifiers']['sentiment']]
    # load topic classifiers
    topic_classifiers = [c['model'].split('/')[-1] if 'AGnews-checkpoint-6000' in c['model'] else c['model'] for c in eval_config['classifiers']['topic']]
    
    sent_all = np.array([eval_preds[c] for c in sent_classifiers]).T
    topic_all = np.array([eval_preds[c] for c in topic_classifiers]).T

    sent_final = sent_all.argmax(-1)
    topic_final = topic_all.argmax(-1)

    sent_max_vote = [sent_all[i][sent_final[i]] for i in range(len(sent_all))]
    topic_max_vote = [topic_all[i][topic_final[i]] for i in range(len(topic_all))]

    combine_pred = ['#'.join([sent_max_vote[i], topic_max_vote[i]]) for i in range(len(sent_max_vote))]

    eval_results['multiple'] = {}
    eval_results['multiple']['overall'] = compute_acc_prec_rec_f1(combine_pred, eval_preds['ground_truth'], average='weighted')

    eval_results['multiple']['sentiment'] = compute_acc_prec_rec_f1(sent_max_vote, [g.split('#')[0] for g in eval_preds['ground_truth']], pos_label='positive', average='binary')
    eval_results['multiple']['topic'] = compute_acc_prec_rec_f1(topic_max_vote, [g.split('#')[1] for g in eval_preds['ground_truth']], average='weighted')

    # calculate metrics for each control value
    sent_labels = load_control_values('sentiment')
    topic_labels = load_control_values('topic')
    for sent_lbl in sent_labels:
        for topic_lbl in topic_labels:
            label = f'{sent_lbl}#{topic_lbl}'
            pred = [combine_pred[i] for i in range(len(combine_pred)) if eval_preds['ground_truth'][i] == label]
            if len(pred) > 0:
                eval_results['multiple'][label] = compute_acc_prec_rec_f1(pred, [label for _ in range(len(pred))], average='weighted')
    save_json(eval_results, eval_res_path)


def calculate_log_probability(input_ids, model):
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    log_probs = [log_prob.sum().item() for log_prob in gen_probs]
    return log_probs


def calculate_slor_batch(sequences, language_model, tokenizer):
    slors = []
    for sequence in sequences:
        input_ids = tokenizer(tokenizer.bos_token+sequence, padding=True, return_tensors="pt").input_ids.to(device)
        log_prob_language_model = calculate_log_probability(input_ids, language_model)
        log_prob_unigram_model = sum(calculate_log_probability(torch.tensor([[50256,input_ids[0][i]] for i in range(1,len(input_ids[0]))]).to(device), language_model) )
        slor = log_prob_language_model[0] - log_prob_unigram_model
        slor /= len(input_ids[0])-1
        slors.append(slor)
    return slors


def evaluate_results(data_df: Dataset, batch_size: int, experiment_name: str, 
                     control_attribute: str, eval_res_path: str) -> Tuple[dict, List[dict]]:

    eval_config = load_json(os.path.join('.', 'scripts', 'src', 'config', 'evaluation.json'))

    classifiers = eval_config['classifiers'][control_attribute.replace('multiple_', '')]
    control_values = load_control_values(control_attribute.replace('multiple_', ''))

    if control_attribute == 'keywords':
        control_values = ['#'.join(c) for c in control_values]

    models_predictions = {}

    if 'multiple' in control_attribute:
        if 'sentiment' in control_attribute:
            index = 0
        else:
            index = 1
        ground_truth = [s.split('#')[index] for s in data_df['train']['control_attribute_value']]
    else:
        ground_truth = data_df['train']['control_attribute_value']
    print("Ground truth: ", len(ground_truth))


    if os.path.exists(eval_res_path):
        eval_results = load_json(eval_res_path)
    else:
        eval_results = {}
    preds_filepath = eval_res_path.replace('evaluation_results', 'evaluation_predictions')
    if os.path.exists(preds_filepath):
        eval_predictions = load_json(preds_filepath)
    else:
        eval_predictions = {'ground_truth': data_df['train']['control_attribute_value']}

    if 'perplexity' not in eval_results or "gpt2-xl" not in eval_results['perplexity']:

        print("Starting calculating perplexity")

        # Initialise results dictionary
        if 'perplexity' not in eval_results:
            eval_results['perplexity'] = {}
        models_predictions['perplexity'] = {}
        eval_results['perplexity'] = {}
        for model in eval_config['perplexity_models']:
            eval_results['perplexity'][model['model']] = {}
            models_predictions['perplexity'][model['model']] = {}
            models_predictions['perplexity'][model['model']]['overall'] = []
            models_predictions['perplexity'][model['model']]['overall_slor'] = []
            models_predictions['perplexity'][model['model']]['overall_unigram'] = []
            for label in control_values:
                models_predictions['perplexity'][model['model']][label] = []

        # Load perplexity metric with HuggingFace evaluate library
        ppl_metric = load_perplexity_model()

        # Load models and tokenizers
        models = {}
        tokenizers = {}
        for model_name in eval_config['perplexity_models']:
            tokenizer, model, _ = load_tokenizer_and_model(model_name)
            models[model_name['model']] = model
            tokenizers[model_name['model']] = tokenizer

        # Compute perplexity for each text
        for index in tqdm(range(0, len(data_df['train']), batch_size),
                        desc=f"Calculating perplexity {experiment_name}"):
            # Get batch of texts
            batch = data_df['train'][index:index + batch_size]

            for model_name in eval_config['perplexity_models']:
                # Calculate perplexity with HuggingFace evaluate library
                perplexity_results = compute_batch_perplexity(batch['text'], model_name['model'],
                                                              ppl_metric, 
                                                              ground_truth[index:index + batch_size], 
                                                              control_values)
                #print(batch['text'])
                slor_values = calculate_slor_batch(batch['text'], models[model_name['model']], 
                                                   tokenizers[model_name['model']])
            
                for label in control_values:
                    if label in perplexity_results:
                        models_predictions['perplexity'][model_name['model']][label].append(perplexity_results[label]['mean_perplexity'])
                models_predictions['perplexity'][model_name['model']]['overall'].append(perplexity_results['overall']['mean_perplexity'])
                models_predictions['perplexity'][model_name['model']]['overall_slor'] += slor_values
                models_predictions['perplexity'][model_name['model']]['overall_unigram'] += [math.exp(-s) for s in slor_values]
        
        for model_name in eval_config['perplexity_models']:
            for label in control_values:
                if len(models_predictions['perplexity'][model_name['model']][label]) > 0:
                    eval_results['perplexity'][model_name['model']][label] = sum(models_predictions['perplexity'][model_name['model']][label])/len(models_predictions['perplexity'][model_name['model']][label])
            eval_results['perplexity'][model_name['model']]['overall'] = sum(models_predictions['perplexity'][model_name['model']]['overall'])/len(models_predictions['perplexity'][model_name['model']]['overall'])
            eval_results['perplexity'][model_name['model']]['overall_slor'] = sum(models_predictions['perplexity'][model_name['model']]['overall_slor'])/len(models_predictions['perplexity'][model_name['model']]['overall_slor'])
            eval_results['perplexity'][model_name['model']]['overall_unigram'] = sum(models_predictions['perplexity'][model_name['model']]['overall_unigram'])/len(models_predictions['perplexity'][model_name['model']]['overall_unigram'])
        save_json(eval_results, eval_res_path)
        print("Finished calculating perplexity")

    for classifier in classifiers:
        model_name = classifier['model']
        if 'Yelp2-checkpoint-64000' in model_name or 'AGnews-checkpoint-6000' in model_name:
            model_name = model_name.split('/')[-1]

        if model_name not in eval_results:
            print(f"Starting computing predictions for {model_name}")
            models_predictions[model_name] = []

            tokenizer, model, classifier = load_tokenizer_and_model(classifier)
            classifier_metrics_res = None
            if model_name == 'Yelp2-checkpoint-64000' or model_name == 'AGnews-checkpoint-6000':
                bin_pred, logits, classifier_metrics_res, wrap_pred = get_batch_prediction_hf_class(data_df['train']['text'], tokenizer,
                                                                                            model, eval_config, model_name, 
                                                                        eval_res_path.replace('evaluation_results', 'processed'),
                                                                        control_attribute, batch_size, 276, classifier)
                
                if wrap_pred is not None:
                    if 'multiple' in control_attribute:
                        g = [c.split('#')[1] for c in data_df['train']['control_attribute_value']]
                    else:
                        g = data_df['train']['control_attribute_value']
                    models_predictions[model_name] = [topics[p] for p in wrap_pred]
                    eval_results[f'{model_name}_binary'] = classifier_metrics_res
                    eval_predictions[f'{model_name}_binary'] = [g[bi] if bin_pred[bi] == 1 else 'null' for bi in range(len(bin_pred))]
                else:
                    models_predictions[model_name] = ['negative' if p == 0 else 'positive' for p in bin_pred]
            else:
                for index in tqdm(range(0, len(data_df['train']), batch_size),
                                desc=f"Calculating predictions {model_name} {experiment_name}"):
                    batch = data_df['train'][index:index + batch_size]
                    
                    predictions, logits, _, _ = get_batch_prediction_hf_class(batch['text'], tokenizer,
                                                                        model, eval_config, model_name, 
                                                                        eval_res_path.replace('evaluation_results', 'processed'),
                                                                        control_attribute.replace('multiple_', ''), batch_size, 276, classifier)

                    models_predictions[model_name] += predictions
                #print(eval_preds)
                #print(models_predictions)
                #a=0/0
            eval_predictions[model_name] = models_predictions[model_name]
            save_json(eval_predictions, preds_filepath)

            eval_results[model_name] = {}
            average = 'binary' if 'sentiment' in control_attribute else 'weighted'
            for label in control_values:
                pred = [models_predictions[model_name][index] for index in range(len(models_predictions[model_name])) if ground_truth[index] == label]
                if len(pred) > 0:
                    eval_results[model_name][label] = compute_acc_prec_rec_f1(pred, [label for _ in range(len(pred))], pos_label=label, average=average)
            label = 'positive' if 'sentiment' in control_attribute else 'business'
            eval_results[model_name]["overall"] = compute_acc_prec_rec_f1(models_predictions[model_name], ground_truth, pos_label=label, average=average)
            save_json(eval_results, eval_res_path)
            print(f"Finished computing predictions for {model_name}")

    if 'control_effectiveness_majority' not in eval_results and (control_attribute == 'sentiment' or control_attribute == 'topic'):
        # Calculate majority voting of classifiers
        print("Starting computing majority voting for control effectiveness")

        eval_results['control_effectiveness_majority'] = {}

        # load classifiers names
        classifiers_names = [c['model'].split('/')[-1] if 'Yelp2-checkpoint-64000' in c['model'] else c['model'] for c in classifiers]
        
        all_res = np.array([eval_predictions[c] for c in classifiers_names]).T
        res_final = all_res.argmax(-1)

        max_vote = [all_res[i][res_final[i]] for i in range(len(all_res))]
        label = 'positive' if 'sentiment' in control_attribute else 'business'
        average = 'binary' if 'sentiment' in control_attribute else 'weighted'

        for label in control_values:
            pred = [max_vote[index] for index in range(len(max_vote)) if ground_truth[index] == label]
            if len(pred) > 0:
                eval_results['control_effectiveness_majority'][label] = compute_acc_prec_rec_f1(pred, [label for _ in range(len(pred))], pos_label=label, average=average)
        
        eval_results['control_effectiveness_majority']['overall'] = compute_acc_prec_rec_f1(max_vote, ground_truth, pos_label=label, average=average)
        save_json(eval_results, eval_res_path)
        print("Finished computing majority voting for control effectiveness")

    if control_attribute == 'keywords' and 'word_coverage' not in eval_results:
        print("Starting computing word coverage")
        eval_results['word_coverage'] = {}

        for label in control_values:
            print(f"Computing word coverage for {label}")
            keywords = []
            texts = []
            for idx in range(len(ground_truth)):
                if ground_truth[idx] == label:
                    keywords.append(ground_truth[idx].split('#'))
                    texts.append(data_df['train']['text'][idx])
            if len(keywords) == 0:
                continue
            soft_coverage, strict_coverage = word_coverage_from_bolt(texts, keywords)
            eval_results['word_coverage'][label] = {
                'soft': soft_coverage,
                'strict': strict_coverage,
                'coverage_score': coverageScore(texts, keywords),
                'original_coverage_score': originalCoverageScore(texts, keywords)
            }
            print(f"Finished computing word coverage for {label}")

        keywords = [k.split('#') for k in ground_truth]
        soft_coverage, strict_coverage = word_coverage_from_bolt(data_df['train']['text'], keywords)
        eval_results['word_coverage']['overall'] = {
            'soft': soft_coverage,
            'strict': strict_coverage,
            'coverage_score': coverageScore(data_df['train']['text'], keywords),
            'original_coverage_score': originalCoverageScore(data_df['train']['text'], keywords)
        }
        save_json(eval_results, eval_res_path)
        print("Finished computing word coverage")

    if 'distinct-n' not in eval_results or 'overall' not in eval_results['distinct-n']:
        print("Starting computing Distinct-n")

        eval_results['distinct-n'] = {'overall': {'dist1': [], 'dist2': [], 'dist3': []}}
        models_predictions['distinct-n'] = {'overall': {'dist1': [], 'dist2': [], 'dist3': []}}

        for label in control_values:
            print(f"Computing distinct-n for {label}")
            eval_results['distinct-n'][label] = {}

            texts = []
            for idx in range(len(ground_truth)):
                if ground_truth[idx] == label:
                    texts.append(data_df['train']['text'][idx])
            if len(texts) == 0:
                continue

            dist1, dist2, dist3 = eval_distinct(texts)
            eval_results['distinct-n'][label]['dist1'] = dist1
            eval_results['distinct-n'][label]['dist2'] = dist2
            eval_results['distinct-n'][label]['dist3'] = dist3

            models_predictions['distinct-n']['overall']['dist1'].append(dist1)
            models_predictions['distinct-n']['overall']['dist2'].append(dist2)
            models_predictions['distinct-n']['overall']['dist3'].append(dist3)

        eval_results['distinct-n']['overall']['dist1'] = sum(models_predictions['distinct-n']['overall']['dist1'])/len(models_predictions['distinct-n']['overall']['dist1'])
        eval_results['distinct-n']['overall']['dist2'] = sum(models_predictions['distinct-n']['overall']['dist2'])/len(models_predictions['distinct-n']['overall']['dist2'])
        eval_results['distinct-n']['overall']['dist3'] = sum(models_predictions['distinct-n']['overall']['dist3'])/len(models_predictions['distinct-n']['overall']['dist3'])
        save_json(eval_results, eval_res_path)
        print("Finished computing Distinct-n")


def remove_empty_texts(examples: dict) -> dict:
    transf_data = {
        "original_id": [],
        "prompt": [],
        "id": [],
        "control_attribute_value": [],
        "text": []
    }
    for index in range(len(examples['prompt'])):
        text = examples['text'][index]
        text = text if text is not None else ' '
        transf_data['id'].append(examples['id'][index])
        transf_data['original_id'].append(examples['original_id'][index])
        transf_data['prompt'].append(examples['prompt'][index])
        transf_data['control_attribute_value'].append(examples['control_attribute_value'][index])
        transf_data['text'].append(text)
    return transf_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder_path', metavar='path',
                        default=os.path.join('.', 'results'))
    parser.add_argument('--control_attribute', type=str, required=True,
                        choices=["sentiment", "topic", "keywords", "multiple"])
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    arguments = parser.parse_args()

    set_seed(42)

    results_path = os.path.join(str(arguments.results_folder_path),
                                str(arguments.control_attribute))

    folder_path = os.path.join(results_path, arguments.folder)

    processed_texts_path = os.path.join(folder_path, f'processed_{arguments.folder}.csv')
    eval_res_path = os.path.join(folder_path, f'evaluation_results_{arguments.folder}.json')

    if os.path.exists(processed_texts_path):
        texts_hf = load_dataset("csv", data_files=processed_texts_path)
        texts_hf = texts_hf.map(remove_empty_texts, batched=True,
                                batch_size=arguments.batch_size)

        print(f'\nLoading evaluation results from {eval_res_path}\n')

        if arguments.control_attribute == 'multiple':
            evaluate_results(texts_hf, arguments.batch_size, arguments.folder,
                            'multiple_sentiment', eval_res_path)
            evaluate_results(texts_hf, arguments.batch_size, arguments.folder,
                            'multiple_topic', eval_res_path)
            compute_multiple_control_metrics(eval_res_path.replace('_results', '_predictions'), eval_res_path)
        else:
            evaluate_results(texts_hf, arguments.batch_size, arguments.folder,
                            arguments.control_attribute, eval_res_path)

        #eval_preds_path = os.path.join(folder_path, f'evaluation_predictions_{arguments.folder}.json')
        #save_json(eval_preds, eval_preds_path)
    else:
        print(f"Processed texts path {processed_texts_path} does not exist.")
