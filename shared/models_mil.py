# # models_mil.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging
# # 我们需要从你已有的 models.py 文件中导入 ComponentViTEncoder
# # 假设你的旧模型文件名为 models.py，并且在同一个目录下
# # 如果不在，你需要调整导入路径，或者直接把 ComponentViTEncoder 类的代码复制到这里
# from models import ComponentViTEncoder 



# class MILClassifier(nn.Module):
#     """
#     【最终版 - RoI-centric】
#     顶层模型。它使用无CLS Token的编码器，并且只有一个实例级别的分类头。
#     """
#     def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, n_classes=2, **kwargs):
#         super().__init__()
        
#         # 编码器现在只输出 RoI 特征
#         self.component_encoder = ComponentViTEncoder(input_dim, hidden_dim, n_heads, n_layers)
        
#         # 只有一个实例级别的分类头
#         self.instance_classifier = nn.Linear(hidden_dim, n_classes)

#     def forward(self, batch: dict):
#         sequences = batch['sequences'] # (Total_Components, MAX_ROIS, Dim)
#         masks = batch['masks']         # (Total_Components, MAX_ROIS) (True is valid)
        
#         # 1. 从编码器获取 RoI 的输出特征
#         # 注意：roi_outputs 的序列长度可能是被压缩过的
#         roi_outputs = self.component_encoder(sequences, masks)
#         # -> shape: (Total_Components, L_compressed, hidden_dim)
        
#         # --- 2. 【核心修复】正确地筛选出所有有效的 RoI token ---
        
#         # a. 获取压缩后的长度
#         B, L_compressed, D = roi_outputs.shape
        
#         # b. 将原始的 padding mask 也裁切到相同的长度
#         masks_compressed = masks[:, :L_compressed] # shape: (Total_Components, L_compressed)
        
#         # c. 使用裁切后的掩码，安全地挑选出所有有效的 RoI token
#         # masks_compressed 作为布尔索引，作用在 roi_outputs 的前两个维度上
#         valid_roi_outputs = roi_outputs[masks_compressed]
#         # -> shape: (Total_Instances_in_Batch, hidden_dim)
        
#         # 3. 对有效的 RoI token 进行分类
#         instance_logits = self.instance_classifier(valid_roi_outputs)
#         # -> shape: (Total_Instances_in_Batch, n_classes)
        
#         # 4. 返回最终的实例级 logits
#         return {"instance_logits": instance_logits}

# models_mil.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# 我们需要从你已有的 models.py 文件中导入 ComponentViTEncoder
# 假设你的旧模型文件名为 models.py，并且在同一个目录下
# 如果不在，你需要调整导入路径，或者直接把 ComponentViTEncoder 类的代码复制到这里
from models import ComponentViTEncoder 

class GatedAttention(nn.Module):
    """
    门控注意力机制模块 (来自 ABMIL 论文)。
    这个模块为一系列的实例嵌入计算注意力权重。
    """
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        # 注意力门的两个分支
        self.attention_V1 = nn.Linear(input_dim, attention_dim)
        self.attention_V2 = nn.Linear(input_dim, attention_dim)
        # 计算标量分数的最终线性层
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, component_cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        为每个实例计算原始的（softmax之前）注意力分数。
        
        Args:
            component_cls_tokens (Tensor): 批次中所有连通分量的 [CLS] token。
                                           形状: (总连通分量数, Hidden_Dim)

        Returns:
            Tensor: 每个连通分量的原始注意力分数。 形状: (总连通分量数, 1)
        """
        # 分支 1: tanh(V1 * h_n)
        tanh_out = torch.tanh(self.attention_V1(component_cls_tokens))
        
        # 分支 2: sigm(V2 * h_n)
        sigmoid_out = torch.sigmoid(self.attention_V2(component_cls_tokens))
        
        # 逐元素相乘
        element_wise_product = tanh_out * sigmoid_out
        
        # 计算最终的标量分数
        raw_attention_scores = self.attention_w(element_wise_product)
        
        return raw_attention_scores


class MILClassifier(nn.Module):
    """
    使用门控注意力机制的完整MIL模型。
    这个模型有两个预测头:
    1. 实例级别: 为每个独立的连通分量预测一个分类结果。
    2. 包级别 (病人级别): 在注意力聚合后，为整个病人预测最终的分类结果。
    """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, n_classes=2, attention_dim=128):
        super().__init__()
        
        # 将每个连通分量编码为一个 [CLS] token
        self.component_encoder = ComponentViTEncoder(input_dim, hidden_dim, n_heads, n_layers)
        # self.bag_classifier = nn.Linear(hidden_dim, n_classes)
        # 预测头
        self.instance_classifier = nn.Linear(hidden_dim, n_classes) # 用于每个连通分量
       

    def forward(self, batch: dict):
        sequences = batch['sequences']
        masks = batch['masks']
        
        # 1. 从编码器获取双轨输出
        encoder_outputs = self.component_encoder(sequences, masks)
        
        cls_outputs = encoder_outputs['cls_output']     # (Total_Components, hidden_dim)
        roi_outputs = encoder_outputs['roi_outputs']     # (Total_Components, MAX_ROIS, hidden_dim)
        
        # --- 2. 处理包级别轨道 ---
        # 直接对 [CLS] token 进行分类，得到每个连通分量的logits
        bag_logits = self.instance_classifier(cls_outputs)
        # bag_logits = self.bag_classifier(cls_outputs)
        # -> shape: (Total_Components, n_classes)
        
        # --- 3. 处理实例级别轨道 ---

        B, L, D = roi_outputs.shape  # L 应该是 MAX_ROIS
        if masks.shape[1] != L:
        # 如果 masks 的第二个维度不等于 roi_outputs 的第二个维度
        # 调整 masks 的形状
            if masks.shape[1] > L:
                masks = masks[:, :L]  # 截断
        else:
            # 如果 masks 较短，需要填充
            padding = torch.zeros(B, L - masks.shape[1], dtype=torch.bool, device=masks.device)
            masks = torch.cat([masks, padding], dim=1)

        # a. 首先，我们需要将所有有效的 RoI token 挑选出来并展平
        # masks shape: (Total_Components, MAX_ROIS)
        # 我们只选择那些 masks 中为 True 的 roi_outputs
        valid_roi_outputs = roi_outputs[masks]
        # -> shape: (Total_Instances_in_Batch, hidden_dim)
        
        # b. 对有效的 RoI token 进行分类
        instance_logits = self.instance_classifier(valid_roi_outputs)
        # -> shape: (Total_Instances_in_Batch, n_classes)
        # logging.info(f'bag_logits{bag_logits.shape}')
        # logging.info(f'instance_logits{instance_logits.shape}')
        # 4. 返回一个包含两个 logits 的字典，供损失函数使用
        return {
            "bag_logits": bag_logits,
            "instance_logits": instance_logits
        }


