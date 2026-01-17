#%%
import torch
from pathlib import Path
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")  # requires sentencepiece

k = 5

# Choose analysis mode
analysis_mode = "vcd_tokens"  # Options: "vcd_tokens" or "layer_scores"

if analysis_mode == "vcd_tokens":
    # Analyze tokens from VCD sampling output
    token_logits_output_file = '/mnt/tmpfs/next_token_logits.pt'  # Path to output from vcd_sample.py
    
    if not Path(token_logits_output_file).exists():
        print(f"Error: Output file {token_logits_output_file} not found.")
        print("Make sure to pass 'token_logits_output_file' in model_kwargs when calling generate().")
    else:
        # Load the saved probability distribution tensor
        probs = torch.load(token_logits_output_file, map_location=torch.device("cpu"), weights_only=True)
        
        print(f"\n{'='*60}")
        print(f"Token Distribution Analysis")
        print(f"{'='*60}")
        print(f"Vocab size: {probs.shape[0]}")
        
        # Get top-k tokens
        top_k_values, top_k_indices = torch.topk(probs, k)
        
        print(f"\nTop {k} tokens:")
        for rank in range(k):
            token_id = top_k_indices[rank].item()
            prob = top_k_values[rank].item()
            
            # Decode the token
            decoded_token = tokenizer.decode([token_id])
            print(f"  Rank {rank + 1}: token_id={token_id:6d}, prob={prob:8.4f}, token={decoded_token!r}")

elif analysis_mode == "layer_scores":
    # Original layer scores analysis
    filepath = '/mnt/tmpfs/output_scores_11.pt'
    is_output_score = True
    #%%
    layer_scores: torch.Tensor = torch.load(
        filepath,
        map_location=torch.device("cpu"),
        weights_only=True
    )
    #dict with layer_idx as key (32+1 layers). #TODO why +1
    #%%
    if is_output_score:
        # Remove batch_size
        batch_size_dim = 0
        batch_size = 1
        assert layer_scores.shape[batch_size_dim] == batch_size, f"Expected dimension {batch_size_dim} to be batch_size, == {batch_size}, but got {layer_scores.shape[batch_size_dim]}"
        layer_scores = layer_scores.squeeze(batch_size_dim)
    #%%
    layer_count = layer_scores.shape[0]

    # Last layers
    layer_scores_last_layers = layer_scores
    #%%
    # Top-k
    top_k_layer_scores = torch.topk(layer_scores_last_layers, k)

    #indices == tokens; values == probabilities
    tokens = top_k_layer_scores.indices
    probs = top_k_layer_scores.values
    #%%
    def print_tokens_at_layer(tokens, probs, tokenizer):
        for token_idx in range(tokens.shape[0]):
            token_id = tokens[token_idx].item()
            prob = probs[token_idx].item()

            # Decode the token
            decoded_token = tokenizer.decode([token_id])

            print(f"  ({decoded_token!r}, {prob:.4f})")

    # Decode and print for each layer
    if is_output_score:
        print_tokens_at_layer(tokens, probs, tokenizer)
    else:
        for layer_idx in reversed(range(tokens.shape[0])):
            print(f"\nLayer {layer_idx-layer_count}:")
            print_tokens_at_layer(tokens[layer_idx], probs[layer_idx], tokenizer)