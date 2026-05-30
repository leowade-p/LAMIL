"""Load DINOv2 ViT-L/14 backbone with optional pretrained weights."""
import torch


def load_dinov2_base_encoder(repo_path: str, weights_path: str, map_location="cpu"):
    encoder = torch.hub.load(repo_path, "dinov2_vitl14_reg", source="local")
    if weights_path:
        state = torch.load(weights_path, map_location=map_location)
        encoder.load_state_dict(state)
    return encoder
