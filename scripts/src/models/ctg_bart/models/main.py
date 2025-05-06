# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 5:45 PM
# @Author  : He Xingwei
import torch
from torch import Tensor
import time
import sys
sys.path.append('../')
from torch.nn.utils.rnn import pad_sequence
from src.models.ctg_bart.models.bart import BARTDataset

# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
torch.manual_seed(0)

# encoder_labels : 0 for copy, 1 for replacement, 2 for insertion
# indicate_labels: 0 for copy, 1 for copy and insertion, 2 for copy, replacement and insertion, 3 for replacement

def generate(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        num_beams = 1,
        temperature=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1,
        refinement_steps=3,
        max_refinement_steps=10,
        adaptive=False,
        show_refine=0,
        threshold=0,
        decoder_chain = 1,
        rank_lm = None,
        max_len = 40
    ):

    bts = len(indicate_labels)
    if do_sample:
        effective_bts = bts*decoder_chain
        if decoder_chain>1:
            # expand inputs
            encoder_inputs = [e.clone() for e in encoder_inputs for i in range(decoder_chain)]
            if decoder_inputs is not None:
                decoder_inputs = [e.clone() for e in decoder_inputs for i in range(decoder_chain)]
            indicate_labels = [e[:] for e in indicate_labels for i in range(decoder_chain)]
    else:
        effective_bts = bts

    batch_refinement_steps = torch.tensor([0] * effective_bts)


    if adaptive:
        current_refinement = 0
        done = False
        while not done:
            predict_outputs, indicate_labels, batch_refinement,decoder_lengths = generate_step_parallel\
                (model, tokenizer, encoder_inputs, indicate_labels, encoder_loss_type,max_insert_label,device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor, sub_tokens_tensor = sub_tokens_tensor, repetition_penalty=repetition_penalty,
                 num_beams=num_beams,temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p,threshold=threshold,
                 max_len = max_len
                 )
            encoder_inputs = predict_outputs

            current_refinement +=1
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0 or current_refinement == max_refinement_steps:
                done = True
            decoder_inputs = None
    else:
        for i in range(refinement_steps):
            predict_outputs, indicate_labels, batch_refinement, decoder_lengths = generate_step_parallel\
                (model, tokenizer, encoder_inputs, indicate_labels, encoder_loss_type,max_insert_label,device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor, sub_tokens_tensor=sub_tokens_tensor, repetition_penalty=repetition_penalty,
                 num_beams=num_beams,temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p,threshold=threshold,
                 max_len=max_len
                 )

            encoder_inputs = predict_outputs
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0:
                break
            else:
                if show_refine:
                    print(f"refinement {i+1}:")
                    for b in range(effective_bts):
                        print(tokenizer.decode(predict_outputs[b].tolist(), clean_up_tokenization_spaces=False))
                        print(tokenizer.convert_ids_to_tokens(predict_outputs[b].tolist()))
            decoder_inputs = None
    predict_outputs = [predict_outputs[i][:length] for i, length in enumerate(decoder_lengths)]
    if do_sample and decoder_chain>1:
        _predict_outputs = []
        _batch_refinement_steps = []
        # use the rank_lm to select the best one from multi decoder chains
        log_ppls, probs = rank_lm.perplexity(input_ids = predict_outputs)
        log_ppls = log_ppls.view([bts, -1])
        indices = torch.argmax(-log_ppls,  dim=-1,keepdim=False)
        for b in range(bts):
            effective_index = b*decoder_chain + indices[b]
            _predict_outputs.append(predict_outputs[effective_index])
            _batch_refinement_steps.append(batch_refinement_steps[effective_index])

        batch_refinement_steps = _batch_refinement_steps
        predict_outputs = _predict_outputs

    return predict_outputs, batch_refinement_steps

def generate_step(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    start = time.time()
    mask_token_id = tokenizer.mask_token_id
    bos_token_id  = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        pre_predict_outputs = encoder_inputs
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value= pad_token_id)
        attention_mask = torch.zeros(encoder_inputs.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        attention_mask = attention_mask.to(device)
        encoder_inputs = encoder_inputs.to(device)
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts = encoder_inputs.shape[0]
        # s = time.time()
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type,indicate_labels,
                                                                 max_insert_label,threshold=threshold,max_len=max_len)

            # step 3: compute decoder_inputs based on encoder_inputs and encoder_labels
            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist(),
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]
        # e = time.time()
        # print(f'decoder inputs : {e-s}')
        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = pad_token_id

        decoder_lengths = [decoder_inputs[i].shape[0] for i in range(bts)]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value = pad_token_id)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] =  eos_token_id
        decoder_inputs = decoder_inputs.to(device)

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask,encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams>1:
            pass
        else:
            predict_outputs = _generate_no_beam_search(decoder_logits,decoder_inputs, bos_token_id, eos_token_id,
                                                       mask_token_id,indicate_labels,decoder_lengths,
                                                       stop_tokens_tensor=stop_tokens_tensor,
                                                       sub_tokens_tensor=sub_tokens_tensor ,
                                                       temperature=temperature,do_sample=do_sample,
                                                       top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty)
        refinement_steps = []
        for i in range(bts):
            if predict_outputs[i].shape[0]==pre_predict_outputs[i].shape[0] and all(predict_outputs[i]==pre_predict_outputs[i]):
                refinement_steps.append(0)
            else:
                refinement_steps.append(1)
        refinement_steps = torch.tensor(refinement_steps)
        # the predict_outputs is regarded as new encoder_inputs
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths


def generate_step_parallel(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    # start = time.time()
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        if isinstance(encoder_inputs, list):
            encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value= pad_token_id)
            encoder_inputs = encoder_inputs.to(device)

        attention_mask = torch.zeros(encoder_inputs.shape,dtype=torch.float32).to(device)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        pre_predict_outputs = encoder_inputs.clone()
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts, seqlen = encoder_inputs.shape
        pre_decoder_lengths = [len(e) for e in indicate_labels]
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type,indicate_labels,
                                                                 max_insert_label,threshold=threshold,max_len=max_len)

            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist()[:pre_decoder_lengths[i]],
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]

        decoder_lengths = [len(e) for e in indicate_labels]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value = pad_token_id)
        decoder_inputs = decoder_inputs.to(device)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] =  eos_token_id

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask,encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams>1:
            pass
        else:
            indicate_labels_tensor = [torch.tensor(e) for e in indicate_labels]
            indicate_labels_tensor = pad_sequence(indicate_labels_tensor, batch_first=True, padding_value = 1000)
            indicate_labels_tensor = indicate_labels_tensor.to(device)
            predict_outputs = _generate_no_beam_search_parallel(
                    decoder_logits,
                    decoder_labels,
                    mask_token_id,
                    indicate_labels_tensor,
                    stop_tokens_tensor=stop_tokens_tensor,
                    sub_tokens_tensor=sub_tokens_tensor,
                    temperature=temperature, do_sample=do_sample,
                    top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
            )

        refinement_steps = torch.zeros(bts).long()
        for i in range(bts):
            length1 = decoder_lengths[i]
            length2 = pre_decoder_lengths[i]
            if length1 != length2:
                refinement_steps[i] = 1
            else:
                if torch.sum(predict_outputs[i, :length1] == pre_predict_outputs[i,:length1],dim=-1) != length1:
                    refinement_steps[i] = 1
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths

def get_encoder_labels(encoder_logits, encoder_loss_type, indicate_labels_list, max_insert_label = 1, threshold=0,
                       max_len = None):
    if encoder_loss_type==0: # classification
        # argmax
        if threshold>0:
            probs = torch.softmax(encoder_logits, dim=-1)
            # encoder_logits[:,:,1:] += 0.7
            _index = probs[:,:,0]>=threshold
            encoder_logits[_index] = 0
            predict_labels = torch.argmax(encoder_logits,dim=-1,keepdim=False)
            predict_labels[_index] = 0
        else:
            predict_labels = torch.argmax(encoder_logits, dim=-1, keepdim=False)
    else: # regression, round and convert the output into torch.Long tensor
        predict_labels = torch.round(encoder_logits).long()

    for i, e in enumerate(indicate_labels_list):
        if len(e)>max_len+2:
            predict_labels[i][predict_labels[i]==2] = 1 #change insert to replace

    bts = encoder_logits.shape[0]
    new_indicate_labels_list = []
    predict_labels_list = []
    for b in range(bts):
        new_indicate_labels = []
        indicate_labels = indicate_labels_list[b]
        for i,e in enumerate(indicate_labels):
            predict_labels[b, i] = min(predict_labels[b, i], max_insert_label + 1)
            if e ==0: # lexical constraints . only copy is allowed.
                if predict_labels[b,i]!=0:
                    predict_labels[b,i] = 0
            elif e==1: # the boundary token of lexical constraints. copy and insert are allowed.
                if predict_labels[b,i]==1: # change replacement to copy
                    predict_labels[b, i] = 0
            elif e ==2: # generated tokens. all actions are allowed.
                pass
            elif e==3: # only replace is allowed.
                if predict_labels[b, i] == 2: # change insertion to replacement
                    predict_labels[b, i] = 1
            else:
                raise ValueError(f'indicate_labels can only be [0,1,2,3].')

            if predict_labels[b,i]>1: # insert
                new_indicate_labels+=[2]*(predict_labels[b,i]-1)
            new_indicate_labels.append(e)
        new_indicate_labels_list.append(new_indicate_labels)
        predict_labels_list.append(predict_labels[b,:len(indicate_labels)])
    return new_indicate_labels_list, predict_labels_list


def _generate_no_beam_search(
            decoder_logits,
            decoder_inputs,
            bos_token_id,
            eos_token_id,
            mask_token_id,
            indicate_labels_list,
            decoder_lengths,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature = 1,
            do_sample = False,
            top_k  = 0,
            top_p = 1.0,
            repetition_penalty= 1
):
    if temperature != 1:
        decoder_logits = decoder_logits / temperature
    bts = decoder_inputs.shape[0]
    new_encoder_inputs_list = []
    for b in range(bts):
        new_encoder_inputs = []
        indicate_labels = indicate_labels_list[b]
        seqlen = decoder_lengths[b]
        for i in range(seqlen):
            if i==0:
                new_encoder_inputs.append(bos_token_id) #
                continue
            if i ==seqlen-1:
                new_encoder_inputs.append(eos_token_id) #
                continue
            if decoder_inputs[b,i+1] == mask_token_id: # we need to predict the mask_token based on the i-th logits
                next_token_logits = decoder_logits[b,i].view(1,-1)
                # set the probability of stop tokens to 0
                if stop_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(stop_tokens_tensor >0 , -1e10)
                # forbid to insert sub tokens behind the lexical constraints
                if i>1 and indicate_labels[i-1]<2 and sub_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(sub_tokens_tensor > 0, -1e10)

                # repetition penalty
                prev_output_tokens = [new_encoder_inputs + decoder_inputs[b].tolist()]
                next_token_logits = enforce_repetition_penalty_\
                    (next_token_logits, 1, 1, prev_output_tokens=prev_output_tokens,
                     repetition_penalty=repetition_penalty)
                if do_sample:
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                new_encoder_inputs.append(next_token.item())
            else:
                new_encoder_inputs.append(decoder_inputs[b,i+1])

        new_encoder_inputs_list.append(torch.tensor(new_encoder_inputs))

    return new_encoder_inputs_list

def _generate_no_beam_search_parallel(
            decoder_logits,
            decoder_labels,
            mask_token_id,
            indicate_labels_tensor,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature = 1,
            do_sample = False,
            top_k  = 0,
            top_p = 1.0,
            repetition_penalty= 1
):
    """
    parallel for batch and seqlen
    :param decoder_logits:
    :param decoder_labels:
    :param mask_token_id:
    :param indicate_labels_tensor:
    :param stop_tokens_tensor:
    :param sub_tokens_tensor:
    :param temperature:
    :param do_sample:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :return:
    """

    if temperature != 1:
        # [b, seq_len, vocab_size]
        decoder_logits = decoder_logits / temperature
    # set the probability of stop tokens to 0
    if stop_tokens_tensor is not None:
        decoder_logits = decoder_logits.masked_fill(stop_tokens_tensor > 0, -1e10)

    # repetition penalty
    decoder_logits = enforce_repetition_penalty_parallel(decoder_logits,
                                                         prev_output_tokens=decoder_labels,
                                                         repetition_penalty=repetition_penalty)

    if sub_tokens_tensor is not None:
        _tmp = indicate_labels_tensor.clone()
        _tmp[:, 1:] = indicate_labels_tensor[:, :-1]
        _tmp[:,1] = 2
        # forbid to insert sub tokens behind the lexical constraints
        lexical_index = _tmp<2
        decoder_logits[lexical_index] = decoder_logits[lexical_index].masked_fill(sub_tokens_tensor > 0, -1e10)
    # predict the mask tokens
    mask_token_index = decoder_labels == mask_token_id
    logits = decoder_logits[mask_token_index]
    if logits.shape[0] == 0 :
        return decoder_labels
    else:
        if do_sample:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            # Sample
            probs = torch.softmax(logits, dim=-1)
            predict_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            predict_tokens = torch.argmax(logits, dim=-1)

        decoder_labels[mask_token_index] = predict_tokens

    return decoder_labels


def _generate_no_beam_search_parallel_v2(
            decoder_logits,
            decoder_labels,
            mask_token_id,
            indicate_labels_tensor,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature = 1,
            do_sample = False,
            top_k  = 0,
            top_p = 1.0,
            repetition_penalty= 1
):
    """
    # the difference between _generate_no_beam_search_parallel and _generate_no_beam_search_parallel_v2 is that:
    prev_output_tokens of the latter method includes last step generated tokens and
    tokens generated in this step before the current token.

     parallel for batch
    :param decoder_logits:
    :param decoder_labels:
    :param mask_token_id:
    :param indicate_labels_tensor:
    :param stop_tokens_tensor:
    :param sub_tokens_tensor:
    :param temperature:
    :param do_sample:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :return:
    """
    if temperature != 1:
        # [b, seq_len, vocab_size]
        decoder_logits = decoder_logits / temperature
    # set the probability of stop tokens to 0
    if stop_tokens_tensor is not None:
        decoder_logits = decoder_logits.masked_fill(stop_tokens_tensor > 0, -1e10)


    if sub_tokens_tensor is not None:
        _tmp = indicate_labels_tensor.clone()
        _tmp[:, 1:] = indicate_labels_tensor[:, :-1]
        _tmp[:,1] = 2
        # forbid to insert sub tokens behind the lexical constraints
        lexical_index = _tmp<2
        decoder_logits[lexical_index] = decoder_logits[lexical_index].masked_fill(sub_tokens_tensor > 0, -1e10)

    seqlen = decoder_labels.shape[1]
    for i in range(seqlen):
        # predict the mask tokens
        logits = decoder_logits[:,i,:]
        mask_token_index = decoder_labels[:,i] == mask_token_id
        logits = logits[mask_token_index]
        if logits.shape[0] == 0:
            continue
        else:
            # repetition penalty
            logits = enforce_repetition_penalty_parallel(logits,
                                                              prev_output_tokens=decoder_labels[mask_token_index],
                                                              repetition_penalty=repetition_penalty)
            if do_sample:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = torch.softmax(logits, dim=-1)
                predict_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                predict_tokens = torch.argmax(logits, dim=-1)

            decoder_labels[:,i][mask_token_index] = predict_tokens
    return decoder_labels

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    elif top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def enforce_repetition_penalty_( lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i]):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty
    return  lprobs

def enforce_repetition_penalty_parallel( lprobs, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    if len(lprobs.shape)==3:
        seqlen = lprobs.shape[1]
        prev_output_tokens = prev_output_tokens.unsqueeze(dim=1).expand(-1,seqlen,-1)
    gather_logits = torch.gather(lprobs, -1, prev_output_tokens)
    gather_logits[gather_logits>0] /= repetition_penalty
    gather_logits[gather_logits < 0] *= repetition_penalty
    lprobs.scatter_(-1, prev_output_tokens, gather_logits)
    return lprobs


