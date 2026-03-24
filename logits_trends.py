#clean up codeand add comments

import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def load_and_categorize_logits(folder_path: str):
    """
    Load all .pt files and categorize logits by prediction type (TP/FP) and condition (old/noisy).
    
    Returns:
        dict: Dictionary with keys like 'TP_old', 'TP_noisy', 'FP_old', 'FP_noisy'
    """
    folder = Path(folder_path)
    pt_files = sorted(folder.glob("*_prediction_*.pt"))
    
    # Store logits by category
    yes_no_logits_list = []
    logits_by_category = defaultdict(lambda: {'base': [], 'noisy': []})
    
    for pt_file in tqdm(pt_files):
        try:
            save_dict = torch.load(pt_file, map_location='cpu')
            logits_base = save_dict['next_token_logits']
            logits_noised = save_dict['next_token_logits_cd']
            # Extract key information
            pred = save_dict['pred'].item() if torch.is_tensor(save_dict['pred']) else save_dict['pred']
            label = save_dict['label'].item() if torch.is_tensor(save_dict['label']) else save_dict['label']

            del save_dict 

            if pred == label and pred in ["yes", "no"]:
                if pred == "yes":
                    prediction_type = "TP"
                else:
                    prediction_type = "TN"
            elif pred != label and pred in ["yes", "no"] and label in ["yes", "no"]:
                if pred == "yes":
                    prediction_type = "FP"
                else:
                    prediction_type = "FN"
            else:
                prediction_type = "Unknown"


            if prediction_type not in logits_by_category:
                logits_by_category[prediction_type] = {'base': [], 'noisy': []}

            pred_token = 3869 if pred == "yes" else 1939 if pred == "no" else None

            # Find the vocab id of the max logit in base
            if torch.is_tensor(logits_base):
                logit_value_base = logits_base.view(-1)[pred_token].item()
                # Get the same token's logit from noised
                if torch.is_tensor(logits_noised):
                    logit_value_noised = logits_noised.view(-1)[pred_token].item()
                else:
                    raise ValueError("logits_noised should be a torch.Tensor")
            else:
                raise ValueError("logits_base should be a torch.Tensor")

            del logits_noised, logits_base 

            if logit_value_noised is not None:
                logits_by_category[prediction_type]['noisy'].append(logit_value_noised)
            if logit_value_base is not None: 
                logits_by_category[prediction_type]['base'].append(logit_value_base)
            
        except Exception as e:
            print(f"Error loading {pt_file.name}: {e}")

    assert len(logits_by_category) > 0, "No logits were loaded. Please check the folder path and file contents."
    for type in logits_by_category.keys():
        print(f"Number of samples for {type}: {len(logits_by_category[type]['base'])} (base), {len(logits_by_category[type]['noisy'])} (noisy)")
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
    
    # Determine bins (same for all histograms)
    all_logits = []
    for logits_by_path in logits_by_category.values():
        all_logits.extend(logits_by_path['base'])
        all_logits.extend(logits_by_path['noisy'])
    
    bins = np.linspace(min(all_logits), max(all_logits), 65)
    
    # Plot each category
    labels = {
        'TP_base': 'TP (base)',
        'TP_noisy': 'TP (noisy)',
        'FP_base': 'FP (base)',
        'FP_noisy': 'FP (noisy)',
    }
    for category in ['FP_base', 'FP_noisy', 'TP_base', 'TP_noisy']:  # Order matters for layering
        type, path = category.split('_')
        if type in logits_by_category and path in logits_by_category[type] and logits_by_category[type][path]:
            ax.hist(logits_by_category[type][path], bins=bins, 
                   label=labels[category], **styles[category])
    ax.set_xlabel('Logit value', fontsize=22)
    ax.set_ylabel('Frequency', fontsize=22)
    ax.legend(loc='upper right', fontsize=22)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")


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
    folder_path = "/Users/manu/Desktop/Projects/Multimodal AI Lab/Mitigate-Relation-Hallucination/Logits/YESNO_shuffle_notransform_42_rerun"

    # Load and categorize the logits
    logits_by_category = load_and_categorize_logits(folder_path)
    
    # Create the plot, calculate statistics
    plot_logit_distribution(logits_by_category, save_path='YESNO_shuffle_notransform_42_rerun.png')