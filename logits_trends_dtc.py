import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math
from collections import Counter


def check_layer_scores(layer_scores):
    """Calculate entropy for each layer score after softmax."""
    entropies = []
    
    # Apply softmax to layer scores
    softmax = nn.Softmax(dim=-1)
    for i, layer_score in enumerate(layer_scores):
        # Convert to tensor if needed
        if not isinstance(layer_score, torch.Tensor):
            layer_score = torch.tensor(layer_score, dtype=torch.float32)
        else:
            layer_score = layer_score.float()
        # Apply softmax
        probs = softmax(layer_score)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        entropies.append(entropy.item())

    # Calculate differences between consecutive entropies
    entropy_diffs = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
    
    # Sort indices based on entropy_diffs values
    sorted_indices = sorted(range(len(entropy_diffs)), key=lambda i: entropy_diffs[i], reverse=True)
    
    return entropies, entropy_diffs, sorted_indices

def load_and_do_vcd(folder_path: str):
    folder = Path(folder_path)
    pt_files = sorted(folder.glob("*layer_scores.pt"))
    counts = Counter()
    # Store logits by category
    logits_by_category = defaultdict(lambda: defaultdict(list))
    sorted_keys_list = []
    for pt_file in tqdm(pt_files):
        try:
            save_dict = torch.load(pt_file, map_location='cpu')
            label = save_dict.get("label", "unknown")
            if save_dict.get("dtc_status") == "enabled":
                counts["dtc_enabled"] += 1
                if save_dict["next_tokens"].item() in [ 3869, 1939]:
                    predicted_token = "yes" if save_dict["next_tokens"].item() == 3869 else "no"
                    if predicted_token == label:
                        counts["correct_withdtc"] += 1
                    else:
                        counts["incorrect_withdtc"] += 1
                        continue
                else:
                    counts["ambiguous_withdtc"] += 1
            else:
                if save_dict["next_tokens"].item() in [ 3869, 1939]:
                    predicted_token = "yes" if save_dict["next_tokens"].item() == 3869 else "no"
                    if predicted_token == label:
                        counts["correct_withoutdtc"] += 1
                        continue
                    else:
                        counts["incorrect_withoutdtc"] += 1
                else:
                    counts["ambiguous_withoutdtc"] += 1
                counts["dtc_disabled"] += 1
            layer_scores = [save_dict[key] for key in sorted(save_dict.keys() - ['next_tokens', 'label','dtc_status', 'yes_no_entropy']) if isinstance(key, int)]
            assert len(layer_scores) == 41, f"No layer scores found in {pt_file}"
            entropies, entropy_diffs, sorted_keys = check_layer_scores(layer_scores)
            sorted_keys_list.append(sorted_keys)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
    print("Counts:", counts)
    
    # Calculate rank based on which index comes at the top
    if sorted_keys_list:
        scores = defaultdict(int)
        rank_1_counts = defaultdict(int)
        for sorted_keys in sorted_keys_list:
            n = len(sorted_keys)
            for i, num in enumerate(sorted_keys):
                scores[num] += (n - i)/n   # earlier = bigger score
            if sorted_keys:
                rank_1_counts[sorted_keys[0]] += 1

        # sort by score descending
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(ranking)
        print("Rank 1 counts:", rank_1_counts)





if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    folder_path = "/home/mt45dumo/runenv/logits/dtc_layer_scores"
    
    load_and_do_vcd(folder_path)