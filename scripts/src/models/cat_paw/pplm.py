#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import numpy as np
from tqdm import trange

import torch
import torch.nn.functional as F
from operator import add
from src.models.cat_paw.style_utils import to_var, top_k_logits

from src.models.cat_paw.classifier.annotator import Attn, MLP
from src.models.cat_paw.gpt2tunediscrim import ClassificationHead


SmallConst = 1e-15

cat_paw_folder = './scripts/src/models/cat_paw/'


def perturb_past(past, model, prev, params, classifier, good_index=None, stepsize=0.01, vocab_size=50257,
                 original_probs=None, accumulated_hidden=None, true_past=None, grad_norms=None, alter_scale=1.0):

    window_length = params['window_length']
    gm_scale, kl_scale = params['fusion_gm_scale'], params['fusion_kl_scale']
    one_hot_vectors = []
    for good_list in good_index:
        good_list = list(filter(lambda x: len(x) <= 1, good_list))
        good_list = torch.tensor(good_list).cuda()
        num_good = good_list.shape[0]
        one_hot_good = torch.zeros(num_good, vocab_size).cuda()
        one_hot_good.scatter_(1, good_list, 1)
        one_hot_vectors.append(one_hot_good)


    # Generate inital perturbed past
    past_perturb_orig = [(np.random.uniform(0.0, 0.0, p.shape).astype('float32'))
                         for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if params['decay']:
        decay_mask = torch.arange(0., 1.0 + SmallConst, 1.0/(window_length))[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, current_length, _ = past[0].shape

    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0].shape[-1:]) #(stack_dim, batch, head, seq_length, head_features) -> (stack_dim, batch, head, window_length, head_features)

        zeros_key_val_shape = tuple(past[0].shape[:-2]) + tuple([current_length - window_length]) + tuple(
            past[0].shape[-1:]) #(stack_dim, batch, head, seq_length, head_features) -> (stack_dim, batch, head, seq_length - window_length, head_features)

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask*ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).cuda() 
    else:
        window_mask = torch.ones_like(past[0]).cuda()

    loss_per_iter = []
    for i in range(params['num_iterations']):
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]

        perturbed_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # Compute hidden using perturbed past
        _, future_past = model(prev, past=perturbed_past)
        hidden = model.hidden_states
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0
        loss_list = []
        if params['loss_type'] == 1 or params['loss_type'] == 3:
            for one_hot_good in one_hot_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word)
                loss_word = -torch.log(loss_word)
                loss += loss_word
                loss_list.append(loss_word)

        if params['loss_type'] == 2 or params['loss_type'] == 3:
            ce_loss = torch.nn.CrossEntropyLoss()
            new_true_past = true_past
            for i in range(params['horizon_length']):

                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)
                
            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + params['horizon_length']))

            # TODO change label class
            label = torch.tensor([params['label_class']], device='cuda', dtype=torch.long)
            discrim_loss = ce_loss(predicted_sentiment, label)
            loss += discrim_loss
            loss_list.append(discrim_loss)


        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (F.softmax(original_probs[:, -1, :], dim=-1))
            p = p + SmallConst * (p <= SmallConst).type(torch.FloatTensor).cuda().detach()
            correction = SmallConst * (probabs <= SmallConst).type(torch.FloatTensor).cuda().detach()
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * ((corrected_probabs * (corrected_probabs / p).log()).sum())
            loss += kl_loss
        
        loss_per_iter.append(loss.data.cpu().numpy())
        loss.backward()
        if grad_norms is not None and params['loss_type'] == 1:
            grad_norms = [torch.max(grad_norms[index], torch.norm(p_.grad * window_mask)) for index, p_ in
                          enumerate(past_perturb)]
        else:
            grad_norms = [(torch.norm(p_.grad * window_mask) + SmallConst) for index, p_ in enumerate(past_perturb)]

        grad = [
            -stepsize * alter_scale * (p_.grad * window_mask / grad_norms[index] ** params['gamma']).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]
    perturbed_past = list(map(add, past, past_perturb))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter


def latent_perturb(model, enc, params, context=None, sample=True, device='cuda'):
    # TODO update discriminator and bow
    # TODO update paths
    bow_index = None
    if params['discrim'] == 'sentiment':
        classifier = ClassificationHead(class_size=5, embed_size=1024).to(device)
        classifier.load_state_dict(torch.load(cat_paw_folder+"discrim_models/sentiment_classifierhead.pt"))
        classifier.eval()
        if params['label_class'] < 0:
            raise Exception('Wrong class for sentiment, use --label-class 2 for *very positive*, 3 for *very negative*')
        if params['activate_alter_scale']:
            if params['annotator_type'] == 'dis':
                if params['classifier_type'] == 'attn':
                    annotator = Attn(1024, 10)
                elif params['classifier_type'] == 'mlp':
                    annotator = MLP(1024, 10)
                annotator.load_state_dict(torch.load(cat_paw_folder+"classifier/Yelp_model_epoch15.pt"))
                annotator.to(device)
                annotator.eval()
            elif params['annotator_type'] == 'bow':
                annotator = None
                bow_index = []
                with open(cat_paw_folder+'wordlists/sentiment_pos.txt', 'r') as f:
                    for line in f.readlines():
                        bow_index.append(enc(line.strip().replace('[SPC]', ' ')).input_ids)
                
                with open(cat_paw_folder+'wordlists/sentiment_neg.txt', 'r') as f:
                    for line in f.readlines():
                        bow_index.append(enc(line.strip().replace('[SPC]', ' ')).input_ids)
        else:
            annotator = None
    else:
        classifier = None
        annotator = None

    # Get tokens for the list of positive words
    def list_tokens(word_list):
        token_list = []
        for word in word_list:
            token_list.append(enc.encode(" " + word))
        return token_list


    good_index = []
    if params['bag_of_words']:
        bags_of_words = params['bag_of_words'].split(";")
        for wordlist in bags_of_words:
            with open(cat_paw_folder+"wordlists/" + wordlist + ".txt", "r") as f:
                words = f.read()
                words = words.split('\n')
            good_index.append(list_tokens(words))
  
    if params['bag_of_words'] and classifier:
        params['loss_type'] = 3

    elif params['bag_of_words']:
        params['loss_type'] = 1

    elif classifier is not None:
        params['loss_type'] = 2

    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')

    if bow_index is not None:
        good_index = [bow_index]

    if params['require_origin']:
        original, _, _ = sample_from_hidden(model=model, params=params, context=context, device=device,
                                    sample=sample, perturb=False, good_index=good_index, classifier=classifier, annotator=annotator)
    torch.cuda.empty_cache()

    perturbed_list = []
    discrim_loss_list = []
    loss_in_time_list = []

    for i in range(params['num_samples']):
        perturbed, discrim_loss, loss_in_time = sample_from_hidden(model=model, params=params, context=context,
                                                         device=device, sample=sample, perturb=True, good_index=good_index,
                                                         classifier=classifier, annotator=annotator)
        perturbed_list.append(perturbed)
        if classifier is not None:
            discrim_loss_list.append(discrim_loss.data.cpu().numpy())
        loss_in_time_list.append(loss_in_time)

    torch.cuda.empty_cache()
        
    if params['require_origin']:
        return original, perturbed_list, discrim_loss_list, loss_in_time_list
    else:
        return perturbed_list, discrim_loss_list, loss_in_time_list


def sample_from_hidden(model, params, classifier, context=None, past=None, device='cuda',
                       sample=True, perturb=True, good_index=None, annotator=None):
    output = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0) if context else None

    def exam_BOW_distribution(good_index, log_probs):
        #good_index = [[input_id1], [inputid2], ...]
        ans = []
        for indices in good_index:

            sum = 0
            for ids in indices:
                
                sum += log_probs[0][ids[0]]

            ans.append(sum.item())
        return ans

    def exam_Disc_distribution(true_hidden, annotator, temperature=0.5):
        probs = F.softmax(annotator(true_hidden)/temperature, dim=-1)[:,-1,:].to('cpu')
        size = probs.shape[-1]
        dist = torch.tensor(range(size)) * (1/size) + (0.5/size)
        if params['discrim'] == 'sentiment':
            res = torch.abs(torch.sum(probs * dist, dim=-1).squeeze() - 0.5).item()

        return res
    

    perplexity = 0.0
    length = 0
    tendency_sit = [0]*len(good_index)

    grad_norms = None
    loss_in_time = []
    for i in trange(params['length'], ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        if past is None and output is not None:
            prev = output[:, -1:]
            _, past = model(output[:, :-1])
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        else:
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        # Modify the past if necessary

        if i >= params['grad_length']:
            current_stepsize = params['stepsize'] * 0
        else:
            current_stepsize = params['stepsize']


        if perturb:
            tmp_original_probs = F.softmax(original_probs[:, -1, :], dim=-1)
            if params['activate_alter_scale'] and params['bag_of_words']:
                alter_scale = np.array(exam_BOW_distribution(good_index, tmp_original_probs)).mean() / params['activesize']

            elif params['activate_alter_scale'] and classifier:
                if params['annotator_type'] == 'dis':
                    alter_scale = exam_Disc_distribution(true_hidden, annotator) / params['activesize']
                elif params['annotator_type'] == 'bow':
                    alter_scale = np.array(exam_BOW_distribution(good_index, tmp_original_probs)).mean() / (2 * params['activesize'])

            else:
                alter_scale = 1.0


        if not perturb or params['num_iterations'] == 0:
            perturbed_past = past

        else:
            accumulated_hidden = model.hidden_states[:, :-1, :]#[bsz, seq_length, dimension]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, prev, params,
                                                                        good_index=good_index, stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        classifier=classifier,
                                                                        grad_norms=grad_norms,
                                                                        alter_scale=alter_scale)
            loss_in_time.append(loss_per_iter)

        test_logits, past = model(prev, past=perturbed_past)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(torch.mean(true_hidden, dim=1))
            label = torch.tensor([params['label_class']], device='cuda', dtype=torch.long)
            true_discrim_loss = ce_loss(predicted_sentiment, label)
        else:
            true_discrim_loss = 0 

        hidden = model.hidden_states  # update hidden
        logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :] / params['temperature']

        log_probs = F.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:
            gm_scale = params['fusion_gm_scale']
            log_probs = ((log_probs ** gm_scale) * (tmp_original_probs ** (1 - gm_scale)))

            log_probs = top_k_logits(log_probs, k=params['top_k'], probs=True)

            if torch.sum(log_probs) <= 1:
                log_probs = log_probs / torch.sum(log_probs)
        
        else:
            logits = top_k_logits(logits, k=params['top_k'])
            log_probs = F.softmax(logits, dim=-1)

        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)

        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
    return output, true_discrim_loss, loss_in_time

