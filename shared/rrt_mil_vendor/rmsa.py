import math

import numpy as np
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def region_partition(x, region_size):
    bsz, height, width, channels = x.shape
    x = x.view(bsz, height // region_size, region_size, width // region_size, region_size, channels)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, channels)
    return regions


def region_reverse(regions, region_size, height, width):
    bsz = int(regions.shape[0] / (height * width / region_size / region_size))
    x = regions.view(bsz, height // region_size, width // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, height, width, -1)
    return x


class InnerAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        epeg=True,
        epeg_k=15,
        epeg_2d=False,
        epeg_bias=True,
        epeg_type="attn",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.epeg_type = epeg_type

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                groups = num_heads if epeg_type == "attn" else head_dim * num_heads
                channels = groups
                self.pe = nn.Conv2d(channels, channels, epeg_k, padding=padding, groups=groups, bias=epeg_bias)
            else:
                groups = num_heads if epeg_type == "attn" else head_dim * num_heads
                channels = groups
                self.pe = nn.Conv2d(
                    channels,
                    channels,
                    (epeg_k, 1),
                    padding=(padding, 0),
                    groups=groups,
                    bias=epeg_bias,
                )
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.pe is not None and self.epeg_type == "attn":
            attn = attn + self.pe(attn)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == "value_bf":
            side = int(np.ceil(np.sqrt(num_tokens)))
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(batch, channels, side, side))
            v = v + pe.reshape(batch, self.num_heads, self.head_dim, num_tokens).permute(0, 1, 3, 2)

        x = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, self.num_heads * self.head_dim)

        if self.pe is not None and self.epeg_type == "value_af":
            side = int(np.ceil(np.sqrt(num_tokens)))
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(batch, channels, side, side))
            x = x + pe.reshape(batch, self.num_heads * self.head_dim, num_tokens).transpose(-1, -2)

        x = self.proj(x)
        return self.proj_drop(x)


class RegionAttntion(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        region_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        region_num=8,
        epeg=False,
        min_region_num=0,
        min_region_ratio=0.0,
        region_attn="native",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == "native":
            self.attn = InnerAttention(
                dim,
                head_dim=head_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                epeg=epeg,
                **kwargs,
            )
        elif region_attn == "ntrans":
            from nystrom_attention import NystromAttention

            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop,
            )
        else:
            raise NotImplementedError(f"Unsupported region_attn: {region_attn}")

    def padding(self, x):
        batch, length, channels = x.shape
        if self.region_size is not None:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % self.region_size
            height, width = height + pad, width + pad
            region_num = int(height // self.region_size)
            region_size = self.region_size
        else:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % self.region_num
            height, width = height + pad, width + pad
            region_size = int(height // self.region_num)
            region_num = self.region_num

        add_length = height * width - length
        if add_length > length / (self.min_region_ratio + 1e-8) or length < self.min_region_num:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % 2
            height, width = height + pad, width + pad
            add_length = height * width - length
            region_size = height

        if add_length > 0:
            x = torch.cat([x, torch.zeros((batch, add_length, channels), device=x.device)], dim=1)
        return x, height, width, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        batch, length, channels = x.shape
        x, height, width, add_length, region_num, region_size = self.padding(x)
        x = x.view(batch, height, width, channels)
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, channels)
        attn_regions = self.attn(x_regions)
        attn_regions = attn_regions.view(-1, region_size, region_size, channels)
        x = region_reverse(attn_regions, region_size, height, width)
        x = x.view(batch, height * width, channels)
        if add_length > 0:
            x = x[:, :-add_length]
        return x


class CrossRegionAttntion(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        region_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        region_num=8,
        epeg=False,
        min_region_num=0,
        min_region_ratio=0.0,
        crmsa_k=3,
        crmsa_mlp=False,
        region_attn="native",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        self.crmsa_mlp = crmsa_mlp

        self.attn = InnerAttention(
            dim,
            head_dim=head_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            epeg=epeg,
            **kwargs,
        )

        if crmsa_mlp:
            self.phi = nn.Sequential(
                nn.Linear(self.dim, self.dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(self.dim // 4, crmsa_k, bias=False),
            )
        else:
            self.phi = nn.Parameter(torch.empty((self.dim, crmsa_k)))
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        batch, length, channels = x.shape
        if self.region_size is not None:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % self.region_size
            height, width = height + pad, width + pad
            region_num = int(height // self.region_size)
            region_size = self.region_size
        else:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % self.region_num
            height, width = height + pad, width + pad
            region_size = int(height // self.region_num)
            region_num = self.region_num

        add_length = height * width - length
        if add_length > length / (self.min_region_ratio + 1e-8) or length < self.min_region_num:
            height = width = int(np.ceil(np.sqrt(length)))
            pad = -height % 2
            height, width = height + pad, width + pad
            add_length = height * width - length
            region_size = height

        if add_length > 0:
            x = torch.cat([x, torch.zeros((batch, add_length, channels), device=x.device)], dim=1)
        return x, height, width, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        batch, length, channels = x.shape
        x, height, width, add_length, region_num, region_size = self.padding(x)
        x = x.view(batch, height, width, channels)
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, channels)

        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)
        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (
            logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8
        )

        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0, 1)
        if return_attn:
            attn_regions, _attn = self.attn(attn_regions, return_attn)
            attn_regions = attn_regions.transpose(0, 1)
        else:
            attn_regions = self.attn(attn_regions).transpose(0, 1)

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)
        attn_regions = attn_regions.view(-1, region_size, region_size, channels)
        x = region_reverse(attn_regions, region_size, height, width)
        x = x.view(batch, height * width, channels)
        if add_length > 0:
            x = x[:, :-add_length]
        return x
