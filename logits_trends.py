#clean up codeand add comments

import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math

def check_effect_of_softmax_on_logits(logits_base, logits_noised):
    """
    Check if applying softmax to logits changes the distribution significantly.
    This is a sanity check to ensure that the logits are in a reasonable range.
    """
    cd_alpha = 1.0
    cd_beta = 0.2
    if not isinstance(logits_base, torch.Tensor) or not isinstance(logits_noised, torch.Tensor):
        print("Logits should be a torch.Tensor")
        return
    if logits_base.numel() == 0 or logits_noised.numel() == 0:
        print("Logits should not be empty")
        return
        
    final_logits_norm = logits_base.clone().float().log_softmax(dim=-1)  # [bsz, vocab]
    cd_logits_norm = logits_noised.clone().float().log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(final_logits_norm, descending=True)
    # cutoff = torch.log(torch.tensor(cd_beta)) + final_logits_norm.max(dim=-1, keepdim=True).values
    cutoff = torch.min(sorted_logits[..., 0], torch.max(final_logits_norm, dim=-1).values  + math.log(cd_beta)).unsqueeze(-1)
    diffs = (1+cd_alpha)*final_logits_norm - cd_alpha*cd_logits_norm
    cd_logits = diffs.masked_fill(final_logits_norm < cutoff, -float("inf"))

    ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding


    next_token_scores = cd_logits
    cd_probs = nn.functional.softmax(cd_logits, dim=-1)
    next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1).item()
    return next_tokens
    
def load_and_do_vcd(folder_path: str):
    folder = Path(folder_path)
    pt_files = sorted(folder.glob("*_prediction_*.pt"))
    correct_predictions = 0
    total_predictions = 0
    ambiguous_predictions = 0
    for pt_file in tqdm(pt_files):
        try:
            save_dict = torch.load(pt_file, map_location='cpu')
            logits_base = save_dict['next_token_logits']
            logits_noised = save_dict['next_token_logits_cd']
            predicted_token = check_effect_of_softmax_on_logits(logits_base, logits_noised)
            predicted_token = "yes" if predicted_token == 3869 else "no" if predicted_token == 1939 else "unknown"
            if predicted_token == save_dict["label"]:
                correct_predictions += 1
            elif predicted_token not in ["yes", "no"]:
                ambiguous_predictions += 1
            total_predictions += 1
        except Exception as e:
            print(f"Error loading {pt_file.name}: {e}")
            continue

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
    print(f"Percentage correct: {correct_predictions / total_predictions * 100 if total_predictions > 0 else 0:.2f}%")
    print(f"Ambiguous predictions: {ambiguous_predictions}")

def load_and_categorize_logits(folder_path: str):
    """
    Load all .pt files and categorize logits by prediction type (TP/FP) and condition (old/noisy).
    
    Returns:
        dict: Dictionary with keys like 'TP_old', 'TP_noisy', 'FP_old', 'FP_noisy'
    """
    folder = Path(folder_path)
    pt_files = sorted(folder.glob("*_prediction_*.pt"))
    
    # Store logits by category
    # yes_no_logits_list = []
    logits_by_category = defaultdict(lambda: {'base': [], 'noisy': []})
    

    for pt_file in tqdm(pt_files):
        try:
            save_dict = torch.load(pt_file, map_location='cpu')
            logits_base = save_dict['next_token_logits']
            logits_noised = save_dict['next_token_logits_cd']
            # Extract key information
            pred = save_dict['pred'].item() if torch.is_tensor(save_dict['pred']) else save_dict['pred']
            label = save_dict['label'].item() if torch.is_tensor(save_dict['label']) else save_dict['label']

            # logits_by_category = {'base': {"yes": logits_base[0,3869], "no": logits_base[0,1939]}, 
            #                         'noisy': {"yes": logits_noised[0,3869], "no": logits_noised[0,1939]},
            #                         'label': label,
            #                         'pred': pred}
            # yes_no_logits_list.append(logits_by_category)
            # continue



            del save_dict  # Free memory

            if pred == label and pred in ["yes", "no"]:
                if pred == "yes":
                    prediction_type = "TP"
                else:
                    prediction_type = "TN"
                # prediction_type = "Correct"  # Combine TP and TN into all correct predictions
            elif pred != label and pred in ["yes", "no"] and label in ["yes", "no"]:
                if pred == "yes":
                    prediction_type = "FP"
                else:
                    prediction_type = "FN"
                # prediction_type = "Incorrect"  # Combine FP and FN into all incorrect predictions
            else:
                prediction_type = "Unknown"
            

            # if prediction_type not in logits_by_category:
            #     logits_by_category[prediction_type] = {}

            
            # Get logit values - adjust based on your needs
            # Using cd_logits, but you might want next_token_logits or next_token_logits_cd
            assert logits_noised.numel() > 0 and logits_base.numel() > 0, "Logits should not be empty"
            # logits_base = nn.functional.softmax(logits_base, dim=-1)
            # logits_noised = nn.functional.softmax(logits_noised, dim=-1)
            logit_value_base = logits_base.max().item() if torch.is_tensor(logits_base) else None
            logit_value_noised = logits_noised.max().item() if torch.is_tensor(logits_noised) else None
            

            del logits_noised, logits_base  # Free memory

            if logit_value_noised is not None:
                logits_by_category[prediction_type]['noisy'].append(logit_value_noised)
            if logit_value_base is not None: 
                logits_by_category[prediction_type]['base'].append(logit_value_base)
            
        except Exception as e:
            print(f"Error loading {pt_file.name}: {e}")
            continue
    
    # with open("yes_no_logits_list.pt", "wb") as f:
    #     torch.save(yes_no_logits_list, f)
    


    assert len(logits_by_category) > 0, "No logits were loaded. Please check the folder path and file contents."
    for type in logits_by_category.keys():
        print(f"NUmber of samples for {type}: {len(logits_by_category[type]['base'])} (base), {len(logits_by_category[type]['noisy'])} (noisy)")


    return logits_by_category


def plot_logit_distribution(logits_by_category, save_path='logit_distribution.png'):
    """
    Create a histogram plot similar to the image showing distribution of logits.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and styles matching the image
    styles = {
        'TP_base': {
            'color': '#6BA3D5', # Blue
            'alpha': 0.7, 
            'histtype': 'stepfilled',
            'edgecolor': 'none'
        },
        'FP_base': {
            'color': '#7ABF7E', # Green
            'alpha': 0.7, 
            'histtype': 'stepfilled',
            'edgecolor': 'none'
        },
        'TP_noisy': {
            'histtype': 'step',
            'color': '#FFA500',  # Orange/Yellow
            'linewidth': 3.0,
            'linestyle': (0, (1, 1)),  # Dotted: (offset, (on, off))
        },
        'FP_noisy': {
            'histtype': 'step',
            'color': '#FF0000',  # Red
            'linewidth': 3.0,
            'linestyle': (0, (1, 1)),  # Dotted
        }
    }
    
    calculate_summary_stats(logits_by_category)
    return
    # Determine bins (same for all histograms)
    all_logits = []
    for logits_by_path in logits_by_category.values():
        all_logits.extend(logits_by_path['base'])
        all_logits.extend(logits_by_path['noisy'])
    
    bins = np.linspace(min(all_logits), max(all_logits), 70)
    
    # Plot each category
    # labels = {
    #     'TP_base': 'TP (base)',
    #     'TP_noisy': 'TP (noisy)',
    #     'FP_base': 'FP (base)',
    #     'FP_noisy': 'FP (noisy)',
    # }
    labels = {
        'Correct_base': 'Correct (base)',
        'Correct_noisy': 'Correct (noisy)',
        'Incorrect_base': 'Incorrect (base)',
        'Incorrect_noisy': 'Incorrect (noisy)',
    }
    style_mapping = {
        'Correct_base': styles['TP_base'],
        'Correct_noisy': styles['TP_noisy'],
        'Incorrect_base': styles['FP_base'],
        'Incorrect_noisy': styles['FP_noisy'],
    }
    #for category in ['FP_base', 'FP_noisy', 'TP_base', 'TP_noisy']:  # Order matters for layering
    for category in ['Correct_base', 'Correct_noisy', 'Incorrect_base', 'Incorrect_noisy']:  
        type, path = category.split('_')
        if type in logits_by_category and path in logits_by_category[type] and logits_by_category[type][path]:
            ax.hist(logits_by_category[type][path], bins=bins, 
                #    label=labels[category], **styles[category])
                label=labels[category], **style_mapping[category])
    ax.set_xlabel('Logit value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()


def calculate_summary_stats(logits_by_category):
    
     # Calculate statistics
    print("\nStatistics:")
    for pred_type,_ in logits_by_category.items():
        for path in ['base', 'noisy']:
            category = f"{pred_type}_{path}"
            logits = logits_by_category[pred_type][path]
            if logits:
                print(f"{category}: n={len(logits)}, mean={np.mean(logits):.3f}, std={np.std(logits):.3f}")
            else:
                print("No logits for category:", category)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    folder_path = "/home/mt45dumo/runenv/logits/YESNO_DINO_Single_rerun"
    
    # load_and_do_vcd(folder_path)
    # exit(0)


    # # Load and categorize the logits
    logits_by_category = load_and_categorize_logits(folder_path)
    
    # # # Create the plot
    # plot_logit_distribution(logits_by_category, save_path='YESNO_DINO_Single_rerun_cor_inc_max.png')