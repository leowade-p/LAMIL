# # models.py

# import math
# from typing import Tuple, List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import config
# import logging
# # =================================================================================
# # Section 1: Segmentation Model and its Helper Classes
# # =================================================================================
# # =================================================================================
# # Section 2: Feature Extractor Wrapper
# # =================================================================================

# class DinoV2FeatureExtractor(nn.Module):
#     """ Your provided DinoV2FeatureExtractor class """
#     def __init__(self, dinov2_model):
#         super().__init__()
#         self.dinov2 = dinov2_model
#         self.feature_dim = self.dinov2.embed_dim
#         # DINOv2 from torch.hub has patch_embed with patch_size as a tuple (H, W)
#         self.patch_size = self.dinov2.patch_embed.patch_size

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         features_dict = self.dinov2.forward_features(x)
#         patch_tokens = features_dict['x_norm_patchtokens']
#         B, C, H, W = x.shape
#         h_feat = H // self.patch_size[0]
#         w_feat = W // self.patch_size[1]
#         feature_map = patch_tokens.reshape(B, h_feat, w_feat, self.feature_dim).permute(0, 3, 1, 2)
#         return feature_map

# # =================================================================================
# # Section 3: Classification Model (Transformer-based)
# # =================================================================================

# class PatchPositionEmbedding(nn.Module):
#     """ Positional embedding for patches within a single RoI feature map. """
#     def __init__(self, grid_size=3, dim=512, learnable=True):
#         super().__init__()
#         self.grid_size = grid_size
#         self.num_patches = grid_size * grid_size
#         self.dim = dim
#         pos_embed = torch.randn(1, self.num_patches, dim) * 0.02
#         if learnable:
#             self.pos_embed = nn.Parameter(pos_embed)
#         else:
#             self.register_buffer('pos_embed', pos_embed)

#     def forward(self, x):
#         return x 

# class ROIBasedTransformerClassifier(nn.Module):
#     """ Downstream Transformer classifier that processes a sequence of RoI tokens. """
#     def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, n_classes=2):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, hidden_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.classifier = nn.Linear(hidden_dim, n_classes)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.global_pos_embed = nn.Parameter(torch.randn(1, config.MAX_ROIS + 1, hidden_dim) * 0.02)

#     def forward(self, x, mask=None):
#         B, L, _ = x.shape
#         cls = self.cls_token.expand(B, -1, -1)
        
#         x = torch.cat([cls, x], dim=1)
#         x = x + self.global_pos_embed
#         if mask is not None:
#             cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
#             full_mask = torch.cat([cls_mask, mask], dim=1)
#             transformer_mask = ~full_mask
#         else:
#             transformer_mask = None
#         encoded_x = self.encoder(x, src_key_padding_mask=transformer_mask)
#         cls_output = self.norm(encoded_x[:, 0])
#         return self.classifier(cls_output)

# class FullTumorClassifier(nn.Module):
#     """ The complete classification model pipeline. """
#     def __init__(self, pos_embedder, transformer_classifier):
#         super().__init__()
#         self.pos_embedder = pos_embedder
#         self.transformer_classifier = transformer_classifier
#         self.tokens_per_roi = pos_embedder.num_patches

#     def forward(self, roi_feat_maps, roi_mask=None):
#         B, MaxRoIs, C, H, W = roi_feat_maps.shape
#         reshaped_maps = roi_feat_maps.reshape(-1, C, H, W)
#         tokens = reshaped_maps.flatten(2).permute(0, 2, 1)
#         tokens_with_pe = self.pos_embedder(tokens)
#         projected_tokens = self.transformer_classifier.proj(tokens_with_pe)
#         final_tokens = projected_tokens.reshape(B, -1, projected_tokens.shape[-1])
#         token_mask = roi_mask.repeat_interleave(self.tokens_per_roi, dim=1) if roi_mask is not None else None
#         return self.transformer_classifier(final_tokens, token_mask)

# class ComponentViTEncoder(nn.Module):
#     """
#     【最终版 - 无CLS Token】
#     这个模块只负责对输入的RoI序列进行Transformer编码，并返回所有RoI的输出特征。
#     """
#     def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, hidden_dim)
        
#         # RoI序列的位置编码，长度为 MAX_ROIS
#         self.pos_embed = nn.Parameter(torch.randn(1, config.MAX_ROIS, hidden_dim) * 0.02)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.1
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, sequences, masks):
#         # sequences: (B, MAX_ROIS, Input_Dim)
#         # masks: (B, MAX_ROIS)  (True 代表有效数据)
        
#         # 1. 投影到隐空间
#         projected_seq = self.proj(sequences)
        
#         # 2. 添加位置编码
#         x = projected_seq + self.pos_embed

#         # 3. 准备 padding mask (PyTorch 中 True 代表“要忽略”)
#         padding_mask = ~masks
        
#         # 4. 通过编码器
#         encoded_x = self.encoder(x, src_key_padding_mask=padding_mask)
        
#         # 5. 归一化并返回所有RoI的输出
#         roi_outputs = self.norm(encoded_x)
        
#         # 【简化】直接返回张量
#         return roi_outputs


# models.py

import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import logging
# =================================================================================
# Section 1: Segmentation Model and its Helper Classes
# =================================================================================
# =================================================================================
# Section 2: Feature Extractor Wrapper
# =================================================================================

class DinoV2FeatureExtractor(nn.Module):
    """ Your provided DinoV2FeatureExtractor class """
    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model
        self.feature_dim = self.dinov2.embed_dim
        # DINOv2 from torch.hub has patch_embed with patch_size as a tuple (H, W)
        self.patch_size = self.dinov2.patch_embed.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_dict = self.dinov2.forward_features(x)
        patch_tokens = features_dict['x_norm_patchtokens']
        B, C, H, W = x.shape
        h_feat = H // self.patch_size[0]
        w_feat = W // self.patch_size[1]
        feature_map = patch_tokens.reshape(B, h_feat, w_feat, self.feature_dim).permute(0, 3, 1, 2)
        return feature_map

# =================================================================================
# Section 3: Classification Model (Transformer-based)
# =================================================================================

class PatchPositionEmbedding(nn.Module):
    """ Positional embedding for patches within a single RoI feature map. """
    def __init__(self, grid_size=3, dim=512, learnable=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.dim = dim
        pos_embed = torch.randn(1, self.num_patches, dim) * 0.02
        if learnable:
            self.pos_embed = nn.Parameter(pos_embed)
        else:
            self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        return x 

class ROIBasedTransformerClassifier(nn.Module):
    """ Downstream Transformer classifier that processes a sequence of RoI tokens. """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.norm = nn.LayerNorm(hidden_dim)
        self.global_pos_embed = nn.Parameter(torch.randn(1, config.MAX_ROIS + 1, hidden_dim) * 0.02)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat([cls, x], dim=1)
        x = x + self.global_pos_embed
        if mask is not None:
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            transformer_mask = ~full_mask
        else:
            transformer_mask = None
        encoded_x = self.encoder(x, src_key_padding_mask=transformer_mask)
        cls_output = self.norm(encoded_x[:, 0])
        return self.classifier(cls_output)

class FullTumorClassifier(nn.Module):
    """ The complete classification model pipeline. """
    def __init__(self, pos_embedder, transformer_classifier):
        super().__init__()
        self.pos_embedder = pos_embedder
        self.transformer_classifier = transformer_classifier
        self.tokens_per_roi = pos_embedder.num_patches

    def forward(self, roi_feat_maps, roi_mask=None):
        B, MaxRoIs, C, H, W = roi_feat_maps.shape
        reshaped_maps = roi_feat_maps.reshape(-1, C, H, W)
        tokens = reshaped_maps.flatten(2).permute(0, 2, 1)
        tokens_with_pe = self.pos_embedder(tokens)
        projected_tokens = self.transformer_classifier.proj(tokens_with_pe)
        final_tokens = projected_tokens.reshape(B, -1, projected_tokens.shape[-1])
        token_mask = roi_mask.repeat_interleave(self.tokens_per_roi, dim=1) if roi_mask is not None else None
        return self.transformer_classifier(final_tokens, token_mask)

class ComponentViTEncoder(nn.Module):
    """
    【新】这个模块是ViT的核心，负责处理一个批次的连通分量序列。
    它为输入的每一个序列（代表一个3D连通分量）计算出一个[CLS] token。
    """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # RoI序列的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, config.MAX_ROIS + 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequences, masks):
        # sequences: (总连通分量数, Max_ROIs, Input_Dim)
        # masks: (总连通分量数, Max_ROIs)

        # 1. 投影到隐空间
        projected_seq = self.proj(sequences)
        B, L, _ = projected_seq.shape # B 在这里是总连通分量数

        # 2. 添加 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, projected_seq], dim=1)
        
        # 3. 添加位置编码
        x = x + self.pos_embed

        # 4. 创建注意力掩码 (Transformer希望True代表被忽略)
        cls_mask = torch.ones((B, 1), dtype=torch.bool, device=masks.device)
        full_mask = torch.cat([cls_mask, masks], dim=1)
        transformer_mask = ~full_mask
        
        # 5. 通过Transformer编码器
        encoded_x = self.encoder(x, src_key_padding_mask=transformer_mask)
        #logging.info(encoded_x.shape)
        # 6. 提取并归一化 [CLS] token
        cls_output = self.norm(encoded_x[:, 0])
        roi_outputs = self.norm(encoded_x[:, 1:])
        return {
            "cls_output": cls_output,
            "roi_outputs": roi_outputs
        }


