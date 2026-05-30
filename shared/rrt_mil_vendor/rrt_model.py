from torch import nn
from timm.models.layers import DropPath

from rrt_mil_vendor.datten import DAttention
from rrt_mil_vendor.rmsa import CrossRegionAttntion, RegionAttntion


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
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


class TransLayer(nn.Module):
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        head=8,
        drop_out=0.1,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        attn="rmsa",
        n_region=8,
        epeg=False,
        region_size=0,
        min_region_num=0,
        min_region_ratio=0,
        qkv_bias=True,
        crmsa_k=3,
        epeg_k=15,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()

        if attn == "ntrans":
            from nystrom_attention import NystromAttention

            self.attn = NystromAttention(
                dim=dim,
                dim_head=trans_dim,
                heads=head,
                num_landmarks=256,
                pinv_iterations=6,
                residual=True,
                dropout=drop_out,
            )
        elif attn == "rmsa":
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs,
            )
        elif attn == "crmsa":
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported attention type: {attn}")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == "gelu" else nn.ReLU
        self.mlp = (
            Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_out)
            if ffn
            else nn.Identity()
        )

    def forward(self, x, need_attn=False):
        x, attn = self.forward_trans(x, need_attn=need_attn)
        if need_attn:
            return x, attn
        return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))
        x = x + self.drop_path(z)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class RRTEncoder(nn.Module):
    def __init__(
        self,
        mlp_dim=512,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="rmsa",
        region_num=8,
        drop_out=0.1,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        epeg=True,
        epeg_k=15,
        region_size=0,
        min_region_num=0,
        min_region_ratio=0,
        qkv_bias=True,
        peg_bias=True,
        peg_1d=False,
        cr_msa=True,
        crmsa_k=3,
        all_shortcut=False,
        crmsa_mlp=False,
        crmsa_heads=8,
        need_init=False,
        **kwargs,
    ):
        super().__init__()
        self.final_dim = mlp_dim
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut
        self.pos_pos = pos_pos
        self.pos_embedding = nn.Identity()

        self.layers = nn.Sequential(
            *[
                TransLayer(
                    dim=mlp_dim,
                    head=n_heads,
                    drop_out=drop_out,
                    drop_path=drop_path,
                    ffn=ffn,
                    ffn_act=ffn_act,
                    mlp_ratio=mlp_ratio,
                    trans_dim=trans_dim,
                    attn=attn,
                    n_region=region_num,
                    epeg=epeg,
                    region_size=region_size,
                    min_region_num=min_region_num,
                    min_region_ratio=min_region_ratio,
                    qkv_bias=qkv_bias,
                    epeg_k=epeg_k,
                    **kwargs,
                )
                for _ in range(max(n_layers - 1, 0))
            ]
        )

        self.cr_msa = (
            TransLayer(
                dim=mlp_dim,
                head=crmsa_heads,
                drop_out=drop_out,
                drop_path=drop_path,
                ffn=ffn,
                ffn_act=ffn_act,
                mlp_ratio=mlp_ratio,
                trans_dim=trans_dim,
                attn="crmsa",
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                crmsa_mlp=crmsa_mlp,
                **kwargs,
            )
            if cr_msa
            else nn.Identity()
        )

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            shape_len = 4

        batch, num_patches, channels = x.shape
        x_shortcut = x

        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        for i, layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)
        if self.all_shortcut:
            x = x + x_shortcut
        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1, 2).reshape(batch, channels, int(num_patches**0.5), int(num_patches**0.5))
        return x


class RRTMIL(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        mlp_dim=512,
        act="relu",
        n_classes=2,
        dropout=0.25,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="rmsa",
        pool="attn",
        region_num=8,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        da_act="relu",
        trans_dropout=0.1,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        da_gated=False,
        da_bias=False,
        da_dropout=False,
        trans_dim=64,
        epeg=True,
        min_region_num=0,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__()
        patch_to_emb = [nn.Linear(input_dim, mlp_dim)]
        if act.lower() == "relu":
            patch_to_emb += [nn.ReLU()]
        elif act.lower() == "gelu":
            patch_to_emb += [nn.GELU()]
        self.patch_to_emb = nn.Sequential(*patch_to_emb)
        self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            pos_pos=pos_pos,
            pos=pos,
            peg_k=peg_k,
            attn=attn,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_path=drop_path,
            drop_out=trans_dropout,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            trans_dim=trans_dim,
            epeg=epeg,
            min_region_num=min_region_num,
            qkv_bias=qkv_bias,
            **kwargs,
        )
        self.pool_fn = (
            DAttention(self.online_encoder.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)
            if pool == "attn"
            else nn.AdaptiveAvgPool1d(1)
        )
        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)
        self.apply(initialize_weights)

    def forward(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        x = self.online_encoder(x)
        if return_attn:
            x, attn = self.pool_fn(x, return_attn=True, no_norm=no_norm)
        else:
            x = self.pool_fn(x)
        logits = self.predictor(x)
        if return_attn:
            return logits, attn
        return logits
