import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from tqdm import tqdm
import json
from sklearn.cluster import KMeans
import random
import numpy as np

from src.models.multi_control.generation_utils import KCenters
from src.models.multi_control.model import AE


pretrained_encoder = "bert-base-uncased"
pretrained_decoder = "gpt2-medium"
latent_size = 768
latent_num = 1
seq_len_per_latent = 20
model_path = './scripts/src/models/multi_control/checkpoint-30000/pytorch_model.bin'
batch_size = 60
max_length = 500
seed = 0

variation = 1e-3

#Parameters for KCenters
num_centers = 1000
num_output_centers = [
        [1,1,5,1],
        [10,10,5,1]
    ]
topk = 200
batch = 5
max_iter = 15
strategy = 'none'
temperature = 50
SDM_reinit = True
weight = {
        "default":[2,7,1],
        "01":[2,4,1],
        "02":[2,8,1],
        "03":[3,1,3],
        "10":[2,12,1],
        "11":[3,5.5,1],
        "12":[2,9,1],
        "13":[3,1,1]
    }

weight = json.loads(weight)


if isinstance(weight, dict):
    default_weight = weight['default']
    weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
    for keys in weight:
        if keys != 'default':
            tmp_i = int(keys[0])
            tmp_j = int(keys[1])
            weight_dict[tmp_i][tmp_j] = weight[keys]
else:
    weight_dict = [[weight for jt in range(4)]for it in range(2)]


if isinstance(num_output_centers, int):
    num_output_centers = [[num_output_centers]*4]*2





encoder_tokenizer = BertTokenizer.from_pretrained(pretrained_encoder)
encoder = BertModel.from_pretrained(pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

model = AE(encoder=encoder, decoder=decoder, args=args)
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device='cuda'

model.to(device)

imdb_dataset = [{'sent':[]} for i in range(2)]
ag_dataset = [{'sent':[]} for i in range(4)]
toxic_dataset = [{'sent':[]} for i in range(2)]

with open('../data/IMDb/IMDb.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        imdb_dataset[label]['sent'].append(line[1].strip())

with open('../data/AGnews/AG-data.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        ag_dataset[label]['sent'].append(line[1].strip())
        label = int(line[0])
        ag_dataset[label]['sent'].append(line[1].strip())


imdb_dataset = [Dataset.from_dict(i) for i in imdb_dataset]
ag_dataset = [Dataset.from_dict(i) for i in ag_dataset]



imdb_dataloader = []
for dataset in imdb_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    imdb_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

ag_dataloader = []
for dataset in ag_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    ag_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))



sentiment_latents = {0:None, 1:None}
topic_latents = {0:None, 1:None, 2:None, 3:None}

for i in range(2):
    for cnt in tqdm(iter(imdb_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']
        
        latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
        if sentiment_latents[i] is None:
            sentiment_latents[i] = latent.squeeze().detach()
        else:
            sentiment_latents[i] = torch.cat((sentiment_latents[i], latent.squeeze().detach()), dim=0)

for i in range(4):
    for cnt in tqdm(iter(ag_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']
        
        latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
        if topic_latents[i] is None:
            topic_latents[i] = latent.squeeze().detach()
        else:
            topic_latents[i] = torch.cat((topic_latents[i], latent.squeeze().detach()), dim=0)



kcmodel = KCenters(num_centers=num_centers, latent_size=latent_size, num_output_centers=num_output_centers, device='cuda')

output_text = []
labels = []



for i in range(2):
    for j in range(4):
        weight = weight_dict[i][j]
        num_output_centers = num_output_centers[i][j]
        print(weight)
        print(num_output_centers)
        centers = kcmodel.train(
            [sentiment_latents[i].to('cuda'), topic_latents[j].to('cuda')],
            weight=weight,
            topk=topk,
            SDM_reinit=SDM_reinit,
            max_iter=max_iter,
            strategy=strategy,
            temperature=temperature,
            num_output_centers=num_output_centers
            ).cpu().numpy()
        centers = [torch.FloatTensor(k).unsqueeze(0) for k in centers]


        for prompts in tqdm(json.loads(pre_tokens)):
            tokens = decoder_tokenizer(prompts, return_tensors='pt')
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            input_ids = input_ids.expand(batch_size, -1)
            attention_mask = attention_mask.expand(batch_size, -1)

            output = model.generate(
                input_latent=random.choice(centers),
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=variation,
                max_len=50,
                rp=1.2
            )

            output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
            labels.extend([[i,j,1]] * batch_size)
            assert len(labels) == len(output_text)
