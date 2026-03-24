
import copy
import inspect
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import transformers
import sys
import os
import json

from torch import nn
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import SampleOutput
from pathlib import Path
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Utils.utils import normalize_to_yesno
from transformers.generation import SampleEncoderDecoderOutput
from transformers.generation.utils import SampleDecoderOnlyOutput

@dataclass
class MyGenerationOutput:
    generation: SampleDecoderOnlyOutput
    attention_cd: torch.Tensor or None

line_counter = 1
counts = Counter()

def aggregate_counts_and_save(pred, label, cd_logits, next_token_logits, next_token_logits_cd, tokenizer, output_folder, experiment_name):
    """
    Saves logits to disk to be used for analysis and debugging

    Args:
        pred: int, predicted token ID
        label: str, ground truth
        cd_logits: Tensor, final logits from contrastive decoding after calibration
        next_token_logits: Tensor, logits from original decoding
        next_token_logits_cd: Tensor, logits from contrastive decoding before calibration
        tokenizer: tokenizer for decoding token IDs
        output_folder: str, folder to save results
        experiment_name: str, name of the experiment for organizing saved results
    """

    global counts

    cd_logits = cd_logits.cpu()
    next_token_logits = next_token_logits.cpu()
    next_token_logits_cd = next_token_logits_cd.cpu()
    
    if label is None:
        # print("Warning: label is None")
        return 
    pred_norm = normalize_to_yesno(tokenizer.decode([pred]))
    label_norm = normalize_to_yesno(label)
    key=""
    if pred_norm is not None and label_norm is not None:
        if pred_norm == label_norm:
            key="correct"
        else:
            key="hallucinated"
    else:
        key="ambiguous"
    
    entropy_before = calculate_entropy(next_token_logits) #all logits
    entropy_after = calculate_entropy(next_token_logits_cd)
    entropy_cd = calculate_entropy(cd_logits)

    save_dict = {
        "pred": pred_norm,
        "label": label_norm,
        "cd_logits": cd_logits,
        "next_token_logits": next_token_logits,
        "next_token_logits_cd": next_token_logits_cd,
        "prediction": key,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_cd": entropy_cd,
        "final_entropy": 1 if entropy_cd > entropy_before else -1
    }
    subkey = "entropy_increased" if entropy_after > entropy_before else "entropy_decreased"
    counts[key+"_"+subkey] += 1

    # Save counts to json file after each aggregation (updated always, immediately flushed)
    counts_output_path = Path(output_folder) / f"aggregate_counts_{experiment_name}.json"
    counts_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(counts_output_path, "w") as f:
        json.dump(counts, f, indent=4)  
    
    #save dict to file 
    output_path = Path(output_folder) / f"{experiment_name}" / f"{line_counter}_prediction_{key}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)


def calculate_entropy(logits: torch.Tensor, k: int = -1) -> float:
    """
    Calculate the entropy of the top-k probabilities.
    
    Args:
        probs: Tensor of shape (vocab_size,)
        k: Number of top probabilities to consider for entropy calculation. -1: all probabilities.
        
    Returns:
        Entropy value of all/top-k probabilities.
    """

    probs = nn.functional.softmax(logits, dim=-1)
    if k > 0:
        # Normalize to sum to 1 after selecting top-k probabilities
        top_k_probs, _ = torch.topk(probs, k)
        top_k_probs = top_k_probs / top_k_probs.sum()  
        entropy = -torch.sum(top_k_probs * torch.log(top_k_probs + 1e-10)).item()  # Adding small value to avoid log(0)
    else:
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return entropy

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    prompt_input_ids = input_ids
    # print("Using patched sample function for VCD...")

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    tokenizer = getattr(self, "_vcd_tokenizer") #Adding tokenizer for logging and debugging, can be safely removed later.
    label = getattr(self.generation_config, "label") #get label from generation config for logging

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    model_kwargs_cd = copy.deepcopy(model_kwargs) # model_kwargs.copy() # copy model_kwargs for cd only for the first forward process
    first_token_generated = False  # Track if we've generated the first token
    output_folder = "/home/mt45dumo/runenv/logits" #model_kwargs.get("token_logits_output_folder", None)  # Optional output folder path
    
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        # Log token distribution for the first token generation
        if not first_token_generated and output_folder is not None:
            global line_counter
            line_counter += 1

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        attention_cd = () if (return_dict_in_generate and output_attentions) else None
        
        if use_cd:
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            
            #model_inputs_cd == model_inputs in all but "images" value
            #print("model_inputs", model_inputs.keys())
            #^doesn't have "images" key, since LlavaLlamaForCausalLM.generate() preprocesses images into inputs_embeds, a key of model_inputs dict
            #print("model_inputs_cd", model_inputs_cd.keys())

            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=output_hidden_states_wo_img,
            )


            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            if attention_cd is not None:
                attention_cd += (
                        (outputs_cd.decoder_attentions,) if self.config.is_encoder_decoder else (outputs_cd.attentions,)
                    )
            
            assert(torch.equal(next_token_logits_cd, next_token_logits) == False)
            
            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = getattr(self.generation_config, "cd_alpha", 1)
            cd_beta = getattr(self.generation_config, "cd_beta", 0.2)
            experiment_name = getattr(self.generation_config, "experiment_name", "default_experiment")
            
            # Log token distribution for CD logits (first token generation only)
            
            # version 1  set cutoff for Adaptive Plausibility Constraints
            # probs = nn.functional.softmax(next_token_logits, dim=-1)
            # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            cd_logits_copy = cd_logits.clone()

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)


            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)



        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,) #nico: [original-vs-noised-attention]: outputs vs outputs_cd
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        # Mark first token as generated after we've logged the logits
        if not first_token_generated and use_cd:
            aggregate_counts_and_save(pred=next_tokens.item(), label=label, cd_logits=cd_logits_copy, next_token_logits=next_token_logits, next_token_logits_cd=next_token_logits_cd,tokenizer=tokenizer, output_folder=output_folder, experiment_name=experiment_name)
        first_token_generated = True
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return MyGenerationOutput(
                    generation=SampleDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                    ),
                    attention_cd=attention_cd
                    
            )
    else:
        return input_ids

def _stash_vcd_to_config(self, kwargs: dict):
    for k in ("cd_alpha", "cd_beta", "label", "experiment_name"):  #Added extra params for debugging and logging.
        if k in kwargs:
            setattr(self.generation_config, f"{k}", kwargs.pop(k))

def evolve_vcd_sampling():
    # print("Patching Transformers sample function for VCD...")
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample

    _orig_generate = transformers.generation.utils.GenerationMixin.generate
    def _generate_patch(self, *args, **kwargs):
        vcd_tk = kwargs.pop("tokenizer")                #Saving tokenizer for logging and debugging
    
        # This line is what creates the attribute on the model
        self._vcd_tokenizer = vcd_tk
        _stash_vcd_to_config(self, kwargs)
        return _orig_generate(self, *args, **kwargs)
    transformers.generation.utils.GenerationMixin.generate = _generate_patch