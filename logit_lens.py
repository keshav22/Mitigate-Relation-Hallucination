import torch

from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b") #requires sentencepiece

n_last_layers = 4
k = 5
filepath = '/mnt/tmpfs/layer_scores_0.pt'
#%%
layer_scores: torch.Tensor = torch.load(
    filepath,
    map_location=torch.device("cpu"),
    weights_only=True
)
#dict with layer_idx as key (32+1 layers). #TODO why +1
#%%
layer_scores = torch.stack([layer_scores[i] for i in sorted(layer_scores)], dim=0) #dict to tensor
#%%
# Remove batch_size
assert layer_scores.shape[1] == 1, f"Expected dimension 1 to be batch_size, == 1, but got {layer_scores.shape[1]}"
layer_scores = layer_scores.squeeze(1)
#%%
layer_count = layer_scores.shape[0]

# Last layers
layer_scores_last_layers = layer_scores[-n_last_layers:, :]
#%%
# Top-k
top_k_layer_scores = torch.topk(layer_scores_last_layers, k)

#indices == tokens; values == probabilities
tokens = top_k_layer_scores.indices
probs = top_k_layer_scores.values
#%%
# Decode and print for each layer
for layer_idx in reversed(range(tokens.shape[0])):
    print(f"\nLayer {layer_count-n_last_layers+1+layer_idx}:")
    for token_idx in range(tokens.shape[1]):
        token_id = tokens[layer_idx, token_idx].item()
        prob = probs[layer_idx, token_idx].item()

        # Decode the token
        decoded_token = tokenizer.decode([token_id])

        print(f"  ({decoded_token!r}, {prob:.4f})")