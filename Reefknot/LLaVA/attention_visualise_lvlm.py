import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import os
from transformers import AutoConfig
from transformers import PreTrainedTokenizer
from typing import Union, Any

@dataclass
class AttentionVisualizer:
    input_ids: torch.Tensor
    image_token_value_in_input_ids: int
    tokenizer: Union[PreTrainedTokenizer, 'PreTrainedTokenizerFast']
    model_path: str
    attentions: Any
    image: "Image.Image"
    image_id: str 
    question_path: str
    attention_start_index: int
    image_token_index: int = -1
    all_generated_tokens: bool = False
    total_image_token_size: int = 576
    total_layers: int = 40
    add_folder_name: str = ""
    
    def __post_init__(self):
        if "VQA" in self.question_path:
            self.all_generated_tokens = True
        
        cfg = AutoConfig.from_pretrained(self.model_path)

        self.total_layers = cfg.num_hidden_layers
        token_image_size = cfg.vision_config.image_size // cfg.vision_config.patch_size
        self.total_image_token_size = token_image_size * token_image_size
        
        for index, idx in enumerate(self.input_ids.tolist()):
            if idx == self.image_token_value_in_input_ids:
                self.image_token_index = index
                break
    
    def visualise_layer_attention_heatmap(self, use_layer_count: int = 1, layer_num: int = -1):
        token_attention = self.attentions[-1]
        
        if self.all_generated_tokens:
            
            selected_layers = range(self.total_layers - use_layer_count, self.total_layers)
            
            token_attention = self.attentions[self.attention_start_index]
            avg_mean_of_attention = torch.stack([
                    token_attention[l].mean(dim=1)  # avg heads
                    for l in selected_layers
                ]).mean(dim=0)[0]
            
            if len(self.attentions) > 2:
                for token_attention in self.attentions[(self.attention_start_index+1):]:
                    avg_mean_of_attention += torch.stack([
                        token_attention[l].mean(dim=1)  # avg heads
                        for l in selected_layers
                    ]).mean(dim=0)[0]
            
                avg_mean_of_attention = avg_mean_of_attention / (len(self.attentions) - 1)
            
            self.save_image_atten_map_plot(avg_mean_of_attention, f'all_generated_token_avg_{use_layer_count}_layers')
        elif use_layer_count > 1:
            selected_layers = range(self.total_layers - use_layer_count, self.total_layers)
            avg_mean_of_attention = torch.stack([
                    token_attention[l].mean(dim=1)  # avg heads
                    for l in selected_layers
                ]).mean(dim=0)[0]
            
            self.save_image_atten_map_plot(avg_mean_of_attention, f'last_{use_layer_count}_layer_avg')
        else:
            last_layer_attention = token_attention[layer_num][0]

            last_layer_attention_tensor = torch.tensor(last_layer_attention)

            avg_mean_of_attention = last_layer_attention_tensor.mean(dim=0)
            
            self.save_image_atten_map_plot(avg_mean_of_attention, "1_layer")
        
    def save_image_atten_map_plot(self, avg_mean_of_attention: torch.Tensor, attention_type: str):
        last_image_token_index = self.image_token_index+self.total_image_token_size
        
        attention_image_on_gen_token = avg_mean_of_attention[0][self.image_token_index:last_image_token_index]
        
        image_attn = attention_image_on_gen_token
        
        img_attn_map = image_attn.reshape(24, 24)

        img_attn_map = img_attn_map - img_attn_map.min()

        img_attn_map = img_attn_map / (img_attn_map.max() + 1e-8)

        attn_map = img_attn_map.unsqueeze(0).unsqueeze(0)  # (1, 1, 24, 24)

        attn_map = F.interpolate(
            attn_map,
            size=(336, 336),
            mode="bilinear",
            align_corners=False
        )

        attn_map = attn_map.squeeze() 

        folder_path = f'./attention_overlay{self.add_folder_name}/{attention_type}'
        file_name = f'{self.image_id}_attention_heatmap.png'
        
        if self.all_generated_tokens:
            folder_path += '/all_generated_tokens' 
        
        os.makedirs(folder_path, exist_ok=True)
        
        save_path = folder_path + '/' + file_name
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.image)
        plt.imshow(attn_map.cpu().numpy(), cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
