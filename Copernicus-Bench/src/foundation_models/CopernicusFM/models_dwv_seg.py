# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from .dynamic_hypernetwork import (
    Dynamic_MLP_OFA_spectral, Dynamic_MLP_OFA_variable
)

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from .aurora.fourier import FourierExpansion
from .flexivit.utils import resize_abs_pos_embed
import math

class CopernicusFMViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=0,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        var_option='language',
        loc_option='lonlat',        
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.patch_embed = Dynamic_MLP_OFA(
        #     wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        # )
        self.patch_embed_spectral = Dynamic_MLP_OFA_spectral(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.patch_embed_variable = Dynamic_MLP_OFA_variable(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim, option=var_option)


        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        #self.patch_embed.num_patches = self.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------
        # prompt setting
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.loc_option = loc_option
        if loc_option == 'cartesian':
            self.coord_expansion = FourierExpansion(1e-7, 2)
        elif loc_option == 'lonlat':
            self.coord_expansion = FourierExpansion(0.0001, 720)
        
        self.scale_expansion = FourierExpansion(0.001, 5.1e8) # 1m2 to 5.1e8 km2
        self.time_expansion = FourierExpansion(1, 365.25, assert_range=False) # 1 to 365.25 days, enable more than 1 year
        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def get_coord_pos_embed(self, lons, lats, embed_dim):
        if self.loc_option == 'cartesian':
            # convert to spherical coordinates
            spherical_x = torch.cos(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_y = torch.sin(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_z = torch.sin(lats * math.pi / 180) + 1 + 1e-7
            coord_embed_spherical_x = self.coord_expansion(spherical_x, embed_dim//3)
            coord_embed_spherical_y = self.coord_expansion(spherical_y, embed_dim//3)
            coord_embed_spherical_z = self.coord_expansion(spherical_z, embed_dim//3)
            coord_embed = torch.cat([coord_embed_spherical_x, coord_embed_spherical_y, coord_embed_spherical_z], dim=-1) # [B,D]
        elif self.loc_option == 'lonlat':
            coord_embed_lon = self.coord_expansion(lons+180, embed_dim//2)
            coord_embed_lat = self.coord_expansion(lats+90, embed_dim//2)
            coord_embed = torch.cat([coord_embed_lon, coord_embed_lat], dim=-1)

        if coord_embed.shape[-1] < embed_dim:
            # pad zeros
            coord_embed = torch.cat((coord_embed, torch.zeros(coord_embed.shape[0], embed_dim-coord_embed.shape[-1], device=coord_embed.device)),dim=-1)

        return coord_embed.unsqueeze(1) # [B,1,D]

    def get_area_pos_embed(self, areas, embed_dim):
        scale_embed = self.scale_expansion(areas, embed_dim) # B, D
        return scale_embed.unsqueeze(1) # [B,1,D]

    def get_time_pos_embed(self, times, embed_dim):
        time_embed = self.time_expansion(times, embed_dim) # B, D
        return time_embed.unsqueeze(1) # [B,1,D]

    def forward_features(self, x, meta_info, key, wave_list, bandwidth, language_embed, input_mode, kernel_size=None):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        bandwidths = torch.tensor(bandwidth, device=x.device).float()
        self.waves = wavelist

        if input_mode == 'spectral':
            x, _ = self.patch_embed_spectral(x, self.waves, bandwidths, kernel_size)
        elif input_mode == 'variable':
            x,waves = self.patch_embed_variable(key, x, self.waves, bandwidths, language_embed, kernel_size)

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        pos_embed = resize_abs_pos_embed(self.pos_embed, num_patches_sqrt, (num_patches_sqrt_origin,num_patches_sqrt_origin), num_prefix_tokens=1)


        # coord, scale and time pos embed
        lons, lats, times, areas = meta_info[:, 0], meta_info[:, 1], meta_info[:, 2], meta_info[:, 3]
        embed_dim = pos_embed.shape[-1]
        if torch.isnan(lons).any() or torch.isnan(lats).any():
            coord_embed = self.coord_token
        else:
            coord_embed = self.get_coord_pos_embed(lons, lats, embed_dim)
        coord_embed = self.coord_fc(coord_embed)
        if torch.isnan(areas).any():
            area_embed = self.scale_token
        else:   
            area_embed = self.get_area_pos_embed(areas, embed_dim)
        area_embed = self.scale_fc(area_embed)
        if torch.isnan(times).any():
            time_embed = self.time_token
        else:
            time_embed = self.get_time_pos_embed(times, embed_dim)
        time_embed = self.time_fc(time_embed)
        pos_embed = pos_embed + coord_embed + area_embed + time_embed


        #hw = self.img_size // self.patch_embed.kernel_size
        hw = num_patches_sqrt
        hw_shape = (hw, hw)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out_features = []

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(out)

        return out_features

    def forward(self, x, meta_info, key, wave_list, bandwidth, language_embed, input_mode, kernel_size=None):
        x = self.forward_features(x, meta_info, key, wave_list, bandwidth, language_embed, input_mode, kernel_size)
        return x


def vit_small_patch16(**kwargs):
    model = CopernicusFMViT(
        out_indices=[3, 5, 7, 11],
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16(**kwargs):
    model = CopernicusFMViT(
        out_indices=[3, 5, 7, 11],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = CopernicusFMViT(
        out_indices=[7, 11, 15, 23],
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = CopernicusFMViT(
        out_indices=[7, 15, 23, 31],
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == "__main__":
    check_point = torch.load(
        "../../../weights/dofa-s-vits-10k.pth"
    )
    vit_model = vit_small_patch16(var_option='language',loc_option='lonlat')
    msg = vit_model.load_state_dict(check_point["model"], strict=False)
    print(msg)

    language_embed = torch.load('var_embed_llama3.2_1B.pt')
    language_embed = language_embed['Sentinel 5P Nitrogen Dioxide']

    x = torch.randn(1,1,56,56)
    key = 's5p_no2'
    wave_list = [0]
    bandwidth = [0]
    input_mode = 'variable'
    meta_info = torch.full((1, 4), float('nan'))
    kernel_size = 4

    y = vit_model(x,key,wave_list,bandwidth,language_embed,input_mode,meta_info,kernel_size)
    print(y[0].shape)
