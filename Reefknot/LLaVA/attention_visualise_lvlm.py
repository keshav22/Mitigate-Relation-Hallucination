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
    attn_bbs: list or None = None
    
    def __post_init__(self):
        if "VQA" in self.question_path:
            self.all_generated_tokens = True
        
        if "Multichoice" in self.question_path:
            self.all_generated_tokens = True
        
        cfg = AutoConfig.from_pretrained(self.model_path)

        self.total_layers = cfg.num_hidden_layers
        token_image_size = cfg.vision_config.image_size // cfg.vision_config.patch_size
        self.total_image_token_size = token_image_size * token_image_size
        
        for index, idx in enumerate(self.input_ids.tolist()):
            if idx == self.image_token_value_in_input_ids:
                self.image_token_index = index
                break

    def get_attention_metric(self, use_layer_count: int = 1):
        selected_layers = range(self.total_layers - use_layer_count, self.total_layers)
        
        token_attention = self.attentions[self.attention_start_index]
        avg_mean_of_attention = torch.stack([
                token_attention[l].mean(dim=1)  # avg heads
                for l in selected_layers
            ]).mean(dim=0)[0]
        
        if len(self.attentions) > self.attention_start_index + 1:
            sequence_length = len(avg_mean_of_attention[0])
            for token_attention in self.attentions[(self.attention_start_index+1):]:
                avg_mean_of_attention += torch.stack([
                    token_attention[l].mean(dim=1)[:, : , :sequence_length]  # avg heads
                    for l in selected_layers
                ]).mean(dim=0)[0]
        
            avg_mean_of_attention = avg_mean_of_attention / (len(self.attentions) - self.attention_start_index)
        
        return self.calculate_attention_metric(avg_mean_of_attention)

    def visualise_layer_attention_heatmap(self, use_layer_count: int = 1, layer_num: int = -1):
        token_attention = self.attentions[-1]
        
        if self.all_generated_tokens:
            
            selected_layers = range(self.total_layers - use_layer_count, self.total_layers)
            
            token_attention = self.attentions[self.attention_start_index]
            avg_mean_of_attention = torch.stack([
                    token_attention[l].mean(dim=1)  # avg heads
                    for l in selected_layers
                ]).mean(dim=0)[0]
            
            if len(self.attentions) > self.attention_start_index + 1:
                sequence_length = len(avg_mean_of_attention[0])
                for token_attention in self.attentions[(self.attention_start_index+1):]:
                    avg_mean_of_attention += torch.stack([
                        token_attention[l].mean(dim=1)[:, : , :sequence_length]  # avg heads
                        for l in selected_layers
                    ]).mean(dim=0)[0]
            
                avg_mean_of_attention = avg_mean_of_attention / (len(self.attentions) - self.attention_start_index)
            
            pre_text = ""
            
            if "VQA" in self.question_path:
                pre_text = "VQA_"
            elif "Multichoice" in self.question_path:
                pre_text = "MCQ_"
            
            self.save_image_atten_map_plot(avg_mean_of_attention, f'{pre_text}all_generated_token_avg_{use_layer_count}_layers')
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
        
        img_attn_map = image_attn.reshape(24, 24) #llava-specific

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
        
        os.makedirs(folder_path, exist_ok=True)
        
        save_path = folder_path + '/' + file_name
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.image)
        plt.imshow(attn_map.cpu().numpy(), cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def calculate_attention_metric(self, avg_mean_of_attention: torch.Tensor):
        last_image_token_index = self.image_token_index+self.total_image_token_size
        
        attention_image_on_gen_token = avg_mean_of_attention[0][self.image_token_index:last_image_token_index]
        
        image_attn = attention_image_on_gen_token
        
        img_attn_map = image_attn.reshape(24, 24) #llava-specific

        weights = self.get_union_weight_matrix(self.attn_bbs)
        
        masked_attention = img_attn_map.cpu() * weights
        total_attention_in_bboxs = masked_attention.sum().item()
        #this metric doesn't care where attended to as long as anywhere in bboxs.
        #IoU better represents that.

        total_image_attention = img_attn_map.sum().item()
        normalized_img_attn = total_attention_in_bboxs / (total_image_attention + 1e-8)

        return normalized_img_attn
    
    def get_union_weight_matrix(self,
        bboxes: list[dict], 
        grid_size: int = 24, #llava-specific
        patch_size: int = 14
    ) -> torch.Tensor:
        """
        Generates a weight matrix for the union of multiple bounding boxes using
        pixel-level masking and average pooling.
        
        Args:
            bboxes: List of dicts {'x', 'y', 'w', 'h'}.
            grid_size: Number of patches per side (default 24).
            patch_size: Pixels per patch (default 14).
            
        Returns:
            torch.Tensor: (grid_size, grid_size) weight matrix.
        """
        # 1. Define full image dimensions
        # 24 * 14 = 336 pixels
        img_dim = grid_size * patch_size
        
        # 2. Create binary pixel mask (False = Background, True = Foreground)
        # Using Boolean ensures overlaps are not double-counted.
        pixel_mask = torch.zeros((img_dim, img_dim), dtype=torch.bool)
        
        for bbox in bboxes:
            # Convert to integer coordinates for slicing
            # We clamp to image dimensions to prevent index-out-of-bounds
            x1 = max(0, int(bbox['x']))
            y1 = max(0, int(bbox['y']))
            x2 = min(img_dim, int(bbox['x'] + bbox['w']))
            y2 = min(img_dim, int(bbox['y'] + bbox['h']))
            
            # Set pixels to True (Boolean OR logic happens naturally here)
            pixel_mask[y1:y2, x1:x2] = True
            
        # 3. Convert to Float for Pooling
        # Shape: (Batch=1, Channel=1, H, W) required for avg_pool2d
        pixel_mask_float = pixel_mask.float().unsqueeze(0).unsqueeze(0)
        
        # 4. Downsample using Average Pooling
        # Kernel size = stride = patch_size ensures we average exactly over each patch
        weight_matrix = F.avg_pool2d(
            pixel_mask_float, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 5. Remove batch/channel dims -> (24, 24)
        return weight_matrix.squeeze()

    def get_bbox_weight_matrix(self,
        bbox: dict, 
        grid_size: int = 24, #llava-specific
        patch_size: int = 14
    ) -> torch.Tensor:
        """
        Generates a weight matrix (24x24) representing the fractional overlap 
        of a bounding box with each patch in the grid.
        
        Args:
            bbox: Dictionary with keys 'x', 'y', 'w', 'h' in absolute pixels.
            grid_size: The number of patches along one dimension (default 24).
            patch_size: The pixel width/height of a single patch (default 14).
            
        Returns:
            torch.Tensor: A (grid_size, grid_size) tensor where each value is 
                        [0.0, 1.0] representing the overlap ratio.
        """
        # 1. unpack bbox coordinates
        bx1 = bbox['x']
        by1 = bbox['y']
        bx2 = bx1 + bbox['w']
        by2 = by1 + bbox['h']

        # 2. Generate patch grid coordinates
        # We create tensors for the left/top edges of all patches
        # shape: (grid_size,)
        patch_indices = torch.arange(grid_size)
        patch_starts = patch_indices * patch_size
        patch_ends = patch_starts + patch_size

        # 3. Create 2D meshes for broadcasting
        # Rows (y-axis): shape (grid_size, 1)
        # Cols (x-axis): shape (1, grid_size)
        p_y1 = patch_starts.unsqueeze(1) 
        p_y2 = patch_ends.unsqueeze(1)
        p_x1 = patch_starts.unsqueeze(0)
        p_x2 = patch_ends.unsqueeze(0)

        # 4. Calculate Intersection Width (Broadcasting x-axis)
        # logic: min(patch_right, bbox_right) - max(patch_left, bbox_left)
        inter_x1 = torch.max(p_x1, torch.tensor(bx1))
        inter_x2 = torch.min(p_x2, torch.tensor(bx2))
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)

        # 5. Calculate Intersection Height (Broadcasting y-axis)
        # logic: min(patch_bottom, bbox_bottom) - max(patch_top, bbox_top)
        inter_y1 = torch.max(p_y1, torch.tensor(by1))
        inter_y2 = torch.min(p_y2, torch.tensor(by2))
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

        # 6. Compute Intersection Area
        # Shape becomes (grid_size, grid_size) via broadcasting
        intersection_area = inter_w * inter_h

        # 7. Normalize by patch area to get weight ratio
        patch_area = patch_size * patch_size
        weight_matrix = intersection_area.float() / patch_area

        return weight_matrix
