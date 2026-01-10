ENABLE_ATTENTION_MAP = False
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
if ENABLE_ATTENTION_MAP:
    import cv2 #attention map
import torch
import torch.distributed as dist
from torch import nn
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from pathlib import Path

from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model

def get_input_ids():
        qs = "Are the signs far from the tractor in this photo? Please answer yes or no." #TODO is hardcoded
        model_path = "/home/as37puta/llava-v1.5-13b" #TODO hardcoded model path
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map="auto")

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["vicuna_v1"].copy() #TODO hardcoded conv mode
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        return input_ids

def save_attention_maps(input_ids, tokenizer, raw_image, output_ids, outputs_attentions, prefix):
    # Calling from this file was just trying out, call from other file instead

    if not ENABLE_ATTENTION_MAP:
        return
    
    #No need, it's already provided as argument:
    #raw_image = Image.open(args.image_file).convert("RGB")
    #image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"][0] #CenterCrop, no padding
    #image_tensor = process_images([raw_image], image_processor, model.config)[0] # padding

    # attentions: list[num_layers] of (batch, num_heads, seq, seq)
    print("att type", type(outputs_attentions)) #att type <class 'tuple'>
    attn_layers = outputs_attentions[-1]  # last step’s attentions
    #print("att layer shape", attn_layers.shape) #att layer shape torch.Size([1, 40, 644, 644])
    num_layers = len(attn_layers)

    # selected_layers = attn_layers[15:28]
    # stacked = torch.stack(selected_layers, dim=0)

    # Average across layers (dim=0), keep heads
    # final_layer_attn = stacked.mean(dim=0).squeeze(0)
    
    final_layer_attn = attn_layers[0][0]  # (num_heads, seq, seq) #edited, was attn_layers[-17][0]
    print("final_layer_attn shape", final_layer_attn.shape) #final_layer_attn shape torch.Size([644, 644])

    # attn_avg = final_layer_attn.mean(dim=0).cpu().numpy()  # (seq, seq) 
    attn_avg, _ = final_layer_attn.max(dim=0)  # (seq_len, seq_len)
    print("attn_avg shape", attn_avg.shape) #attn_avg shape torch.Size([644])
    attn_avg = attn_avg.cpu().numpy()

    full_seq_len = attn_avg.shape[1] #IndexError: tuple index out of range
    input_len = input_ids.shape[1]  
    gen_len = output_ids.shape[1] - input_len

    image_token_count = 576
    print(f"seq_len={full_seq_len}, input_len={input_len}, gen_len={gen_len}, image_tokens={image_token_count}")

    # choose last generated token
    #query_idx = input_len + gen_len - 1 #last token, i.e. </s>
    query_idx = input_len + gen_len - 2
    query_token = tokenizer.decode(output_ids[0, query_idx])

    # text attention
    # Number of image tokens


    # Number of text tokens before <image>
    system_len = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)

    # Number of text tokens after <image>
    user_len = input_len - system_len - 1  # subtract <image> placeholder

    # Number of generated tokens
    gen_len = output_ids.shape[1] - input_len


    image_idx = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)

    # Text tokens before <image>
    text_attn_before = attn_avg[0, 0:image_idx]

    text_start_after = system_len -1 + image_token_count  # keys start after image tokens
    text_end_after = text_start_after + (input_len - image_idx - 1) + gen_len  # user prompt + generated

    text_attn_after = attn_avg[0, text_start_after:689]

    # Combine text attention
    text_attn = np.concatenate([text_attn_before, text_attn_after])

    # Tokens for plotting
    text_ids = input_ids[0].tolist()
    # remove <image> token placeholder
    text_ids.pop(system_len)
    # append generated tokens
    text_ids += output_ids[0, input_len:].tolist()

    text_tokens = [tokenizer.convert_ids_to_tokens(i) for i in text_ids]


    plt.figure(figsize=(20,6))  # wider and taller figure
    #plt.bar(range(len(text_tokens)), text_attn, width=0.8, color='skyblue') #TODO throwing error

    # X-axis labels: tokens, rotated and small font
    plt.xticks(range(len(text_tokens)), text_tokens, rotation=90, fontsize=6)

    plt.ylabel("Attention")
    plt.title(f"Attention on previous text tokens (for '{query_token}')", fontsize=12)
    plt.tight_layout()

    # Show and save
    plt.show()
    #plt.savefig(prefix+"attention_text_skate.jpg", dpi=300) #works. but not needed atm


    # image attention heatmap
    if image_token_count > 0:
        img_start = image_idx 
        img_end = img_start + image_token_count
        img_attn = attn_avg[0,  img_start:img_end]
        img_map = img_attn.reshape(24, 24)

        plt.figure(figsize=(5,5))
        plt.imshow(img_map, cmap="viridis")
        plt.colorbar()
        plt.title(f"Image attention heatmap (for '{query_token}')")
        plt.axis("off")
        plt.show()
        #plt.savefig(prefix+"attention_image_skate.jpg") #works. but not needed atm

        
        # img_map: attention 24x24
        img_map = np.array(img_map)  # ensure numpy
        img_map_norm = (img_map - img_map.min()) / (img_map.max() - img_map.min() + 1e-8)
        img_map_norm = img_map_norm.astype(np.float32)

        # Original image as numpy
        img_np = np.array(raw_image.convert("RGB"))

        # Resize attention map to image size
        img_map_resized = cv2.resize(img_map_norm, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Plot overlay
        plt.figure(figsize=(8,8))
        plt.imshow(img_np)
        plt.imshow(img_map_resized, cmap="viridis", alpha=0.5)
        plt.axis("off")
        plt.title(f"Attention overlay on image (for '{query_token}')")
        plt.tight_layout()
        plt.show()
        plt.savefig(prefix+"attention_image_overlay_skate.jpg", dpi=300)


def _save_token_distribution(logits: torch.Tensor, output_folder: str, label: str, k: int = 5):
    """
    Save the entire probability distribution tensor from logits.
    Assumes batch_size=1.
    
    Args:
        logits: Tensor of shape (1, vocab_size)
        output_folder: Path to folder where results will be saved (directory will be created if needed)
        label: Label for this set of tokens (e.g., "next_token_logits" or "next_token_logits_cd")
        k: Number of top tokens to extract for console output only
    """
    # Move to CPU for processing
    logits = logits.cpu()
    
    # Assert batch_size == 1
    assert logits.shape[0] == 1, f"Expected batch_size=1, but got {logits.shape[0]}"
    
    # Convert logits to probability distribution
    probs = nn.functional.softmax(logits[0], dim=-1)
    label += "_softmaxed" # to avoid any confusion
    
    # Save the full probability distribution tensor
    output_path = Path(output_folder) / f"{label}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probs, output_path)
    
    # Print top-k to console for quick inspection
    print(f"\n{label}:")
    top_k_values, top_k_indices = torch.topk(probs, k)
    for rank in range(k):
        token_id = top_k_indices[rank].item()
        prob = top_k_values[rank].item()
        print(f"  Rank {rank + 1}: token_id={token_id}, prob={prob:.4f}")


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
    model_kwargs_cd = model_kwargs.copy() # copy model_kwargs for cd only for the first forward process
    first_token_generated = False  # Track if we've generated the first token
    output_folder = "/home/nl97naca/run_env" #model_kwargs.get("token_logits_output_folder", None)  # Optional output folder path
    
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
            _save_token_distribution(next_token_logits, output_folder, "next_token_logits", k=5)

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        

        if use_cd:
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            
            #model_inputs_cd == model_inputs in all but "images" value
            print("model_inputs", model_inputs.keys())
            #^doesn't have "images" key, since LlavaLlamaForCausalLM.generate() preprocesses images into inputs_embeds, a key of model_inputs dict
            print("model_inputs_cd", model_inputs_cd.keys())

            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=output_hidden_states_wo_img,
            )
            print("images_cd", type(model_kwargs.get("images_cd")))
            # save_attention_maps(
            #     input_ids=get_input_ids(), #TODO or input_ids, or model_inputs_cd
            #     tokenizer=model_kwargs.get("tokenizer"),
            #     raw_image=model_kwargs.get("images_cd"),
            #     output_ids=outputs_cd.logits.argmax(-1), #TODO or outputs_cd.sequences
            #     outputs_attentions=outputs_cd.attentions
            # )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            assert(torch.equal(next_token_logits_cd, next_token_logits) == False)
            
            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = getattr(self.generation_config, "cd_alpha", 0.5)
            cd_beta = getattr(self.generation_config, "cd_beta", 0.1)
            
            # Log token distribution for CD logits (first token generation only)
            if not first_token_generated and output_folder is not None:
                _save_token_distribution(next_token_logits_cd, output_folder, "next_token_logits_cd", k=5)
            
            # version 1  set cutoff for Adaptive Plausibility Constraints
            # probs = nn.functional.softmax(next_token_logits, dim=-1)
            # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            # Log token distribution for CD probs (first token generation only)
            if not first_token_generated and output_folder is not None:
                # Remove batch dimension for saving
                if cd_logits.shape[0] == 1:
                    cd_logits_output = cd_logits[0]
                else:
                    print(f"Warning: Expected batch_size=1 for cd_logits, but got {cd_logits.shape[0]}")
                    cd_logits_output = cd_logits[0]
                _save_token_distribution(cd_logits_output.unsqueeze(0), output_folder, "cd_logits", k=5)

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

def _stash_vcd_to_config(self, kwargs: dict):
    for k in ("cd_alpha", "cd_beta"):
        if k in kwargs:
            setattr(self.generation_config, f"{k}", kwargs.pop(k))

def evolve_vcd_sampling():
    # print("Patching Transformers sample function for VCD...")
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample

    #transformers.generation.utils.GenerationMixin._validate_model_kwargs = patched_validate_model_kwargs
    _orig_generate = transformers.generation.utils.GenerationMixin.generate
    def _generate_patch(self, *args, **kwargs):
        _stash_vcd_to_config(self, kwargs)
        return _orig_generate(self, *args, **kwargs)
    transformers.generation.utils.GenerationMixin.generate = _generate_patch