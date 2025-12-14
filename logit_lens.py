#%%
import torch

from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b") #requires sentencepiece

k = 5
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
    batch_size_dim=0
    batch_size=1
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