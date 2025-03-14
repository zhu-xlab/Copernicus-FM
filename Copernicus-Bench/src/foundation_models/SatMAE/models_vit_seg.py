# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from util.pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, out_indices, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.out_indices = out_indices

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        out_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x[:, 1:]  # 432 = 144 * 3
                _, hw, D = out.shape
                H_shape = W_shape = int(math.sqrt(hw))
                out = (
                    out.reshape(B, H_shape, W_shape, D).permute(0, 3, 1, 2).contiguous()
                )
                out_features.append(out)

        return out_features


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        out_indices=[3, 5, 7, 11],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        out_indices=[9, 11, 15, 23],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
