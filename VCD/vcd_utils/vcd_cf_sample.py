import warnings
from typing import List, Optional, Union
import torch
import os
import torch.distributed as dist
from torch import nn
import transformers
import gc

from transformers.generation.utils import (
    SampleOutput,
    SampleDecoderOnlyOutput,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation import SampleEncoderDecoderOutput

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

    # debug_logits = bool(model_kwargs.pop("debug_logits", False))

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

    # vcd-cf Parameters
    use_cd = model_kwargs.get("images_cd") is not None
    model_kwargs_cd = dict(model_kwargs)
    cf_input_ids = model_kwargs.pop("cf_input_ids", None)
    use_cf = cf_input_ids is not None and len(cf_input_ids) > 0
    cd_gamma = model_kwargs.get("cd_gamma", 0.5)
    cd_alpha = model_kwargs.get("cd_alpha", 0.5)
    cd_beta = model_kwargs.get("cd_beta", 0.1)
    cf_images = model_kwargs.pop("cf_images", None)
    cf_image_sizes = model_kwargs.pop("cf_image_sizes", None)

    if use_cf:
        cf_static = {"images": cf_images, "image_sizes": cf_image_sizes, "use_cache": False }
        # cf_dyn = [dict() for _ in range(len(cf_input_ids))]
        cf_input_ids = [t.to(input_ids.device) for t in cf_input_ids]
    
    def topk_tokens(logits_2d: torch.Tensor, k: int = 5):
        # logits_2d: (B, V)
        vals, idx = torch.topk(logits_2d[0], k)
        toks = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            toks = self.tokenizer.convert_ids_to_tokens(idx.tolist())
        else:
            toks = [str(i) for i in idx.tolist()]
        return list(zip(toks, [float(v) for v in vals]))
    step = 0

    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0: break

        # base forward pass
        with torch.no_grad():
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            base_logits = outputs.logits[:, -1, :].clone()
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            del outputs, model_inputs

        # noised image forward pass
        img_cd_logits = None
        if use_cd:
            with torch.no_grad():
                model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
                outputs_cd = self(**model_inputs_cd, return_dict=True)
                img_cd_logits = outputs_cd.logits[:, -1, :].clone()
                model_kwargs_cd = self._update_model_kwargs_for_generation(outputs_cd, model_kwargs_cd)
                del outputs_cd, model_inputs_cd

        # cf forward pass

        cf_logits = None
        if use_cf:
            for i in range(len(cf_input_ids)):
                with torch.no_grad():
                    merged_kwargs = {**cf_static}
                    model_inputs_cf = self.prepare_inputs_for_generation(cf_input_ids[i], **merged_kwargs)
                    outputs_cf = self(**model_inputs_cf, return_dict=True)
                    
                    cur_logits = outputs_cf.logits[:, -1, :].clone()
                    updated_kwargs = self._update_model_kwargs_for_generation(outputs_cf, merged_kwargs)
                    del outputs_cf, model_inputs_cf

                    if cf_logits is None:
                        cf_logits = cur_logits
                    else:
                        torch.max(cf_logits, cur_logits, out=cf_logits)
                        del cur_logits

                    torch.cuda.empty_cache() 

        # combine logits
        with torch.no_grad():
            s = 1.0 + (cd_alpha if use_cd else 0.0) + (cd_gamma if use_cf else 0.0)
            log_beta = torch.log(torch.tensor(cd_beta, device=input_ids.device, dtype=base_logits.dtype))
            cutoff = base_logits.max(dim=-1, keepdim=True).values + log_beta 
 
            combined = s * base_logits
            if use_cd: combined -= cd_alpha * img_cd_logits
            if use_cf: combined -= cd_gamma * cf_logits

            combined.masked_fill_(base_logits < cutoff, -float("inf"))

            next_token_scores = logits_processor(input_ids, combined)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores,)

        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if use_cf:
            for i in range(len(cf_input_ids)):
                cf_input_ids[i] = torch.cat([cf_input_ids[i], next_tokens[:, None]], dim=-1)

        # cleanup logits
        del base_logits, img_cd_logits, cf_logits, combined

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            if unfinished_sequences.max() == 0: this_peer_finished = True

        if stopping_criteria(input_ids, scores) or (this_peer_finished and not synced_gpus):
            break
        step += 1

    if streamer is not None:
        streamer.end()
    
    gc.collect()
    torch.cuda.empty_cache()

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
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def patched_validate_model_kwargs(self, model_kwargs):
    return model_kwargs

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._sample = sample
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = patched_validate_model_kwargs
