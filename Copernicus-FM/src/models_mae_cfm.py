# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Yi Wang.
# All rights reserved.

# This code is adapted from https://github.com/facebookresearch/mae, which is 
# licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file except in compliance with the License.
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

from dynamic_hypernetwork_pretrain import Dynamic_MLP_OFA, Dynamic_MLP_Decoder, Dynamic_MLP_OFA_spectral, Dynamic_MLP_OFA_variable
from flexivit.utils import resize_abs_pos_embed
import math
from aurora.fourier import FourierExpansion
import pdb 
import random

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 var_option='language', pos_option='lonlat', time_option='absolute', scale_option='augarea',distill_size='base'):
        super().__init__()

        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # dynamic patch embed with hypernetwork
        self.patch_embed_spectral = Dynamic_MLP_OFA_spectral(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.patch_embed_variable = Dynamic_MLP_OFA_variable(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim, option=var_option)
        #num_patches = self.patch_embed.num_patches
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.waves = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # metadata as pos embed
        self.pos_option = pos_option
        self.time_option = time_option
        self.scale_option = scale_option
        if self.pos_option == 'lonlat':
            self.coord_expansion = FourierExpansion(0.0001, 720) # range of lon/lat coordinates
        elif self.pos_option == 'cartesian':
            self.coord_expansion = FourierExpansion(1e-7, 2) # range of spherical coordinates
        
        self.scale_expansion = FourierExpansion(0.001, 5.1e8) # 0.1km2 to 5.1e8 km2
        self.time_expansion = FourierExpansion(1, 365.25, assert_range=False) # 1 to 365.25 days
        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # dynamic decoder with hypernetwork
        self.decoder_pred_spectral = Dynamic_MLP_Decoder(wv_planes=128, inter_dim=128, kernel_size=16, decoder_embed=decoder_embed_dim)
        self.decoder_pred_variable = Dynamic_MLP_Decoder(wv_planes=128, inter_dim=128, kernel_size=16, decoder_embed=decoder_embed_dim)
        # --------------------------------------------------------------------------
        self.coord_fc_decoder = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.scale_fc_decoder = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.time_fc_decoder = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.scale_token_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.time_token_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.norm_pix_loss = norm_pix_loss

        # distillation from dinov2
        self.teacher_alpha = 0.1
        self.cos = nn.CosineSimilarity(dim=1)
        self.teacher_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.distill_size = distill_size
        if distill_size == 'base':
            distill_dim = 768
        elif distill_size == 'large':
            distill_dim = 1024
        self.student_proj = torch.nn.Linear(embed_dim, distill_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Detect device from existing parameters
        device = 'cpu'  # default to CPU
        if hasattr(self, 'cls_token') and self.cls_token.device.type != 'cpu':
            device = self.cls_token.device
        
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        torch.nn.init.normal_(self.coord_token, std=0.02)
        torch.nn.init.normal_(self.scale_token, std=0.02)
        torch.nn.init.normal_(self.time_token, std=0.02)

        torch.nn.init.normal_(self.coord_token_dec, std=0.02)
        torch.nn.init.normal_(self.scale_token_dec, std=0.02)
        torch.nn.init.normal_(self.time_token_dec, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # load pretrained hypernetwork weights from DOFA
        wg_weights = torch.load('pretrained/weight_generator_1000_0.01_er50k.pt', map_location=device)
        self.patch_embed_spectral.weight_generator.load_state_dict(wg_weights['weight_generator'])
        self.patch_embed_spectral.fclayer.load_state_dict(wg_weights['fclayer'])
        self.patch_embed_variable.weight_generator.load_state_dict(wg_weights['weight_generator'])
        self.patch_embed_variable.fclayer.load_state_dict(wg_weights['fclayer'])

        # load dinov2 from torch hub
        if self.distill_size == 'base':
            self.teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif self.distill_size == 'large':
            self.teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if patch_size is None:
            p = self.patch_embed_spectral.patch_size[0] # default patch size is 16
        else:
            p = patch_size # dynamic patch size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        in_chans = imgs.shape[1]

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_chans))
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if patch_size is None:
            p = self.patch_embed_spectral.patch_size[0] # default patch size is 16
        else:
            p = patch_size # dynamic patch size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        in_chans = x.shape[2] / (p**2)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
        return imgs


    def random_select_channels(self, imgs, wave_list, bandwidth):
        """
        imgs: input origginal
        return: new_imgs with randomly selected channels, wave_list, bandwidth
        75% to 100% channels are randomly selected
        """
        batch_size, num_channels, height, width = imgs.shape
        num_selected_channels = random.randint(int(num_channels*0.75)+1,num_channels)
        selected_indices = torch.randperm(num_channels)[:num_selected_channels]
        selected_channels = imgs[:, selected_indices, :, :]
        nwave_list = [wave_list[int(it)] for it in selected_indices]
        bandwidth = [bandwidth[int(it)] for it in selected_indices]
        return selected_channels, nwave_list, bandwidth


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_coord_pos_embed(self, lons, lats, embed_dim):
        if self.pos_option=='cartesian':
            # convert to spherical coordinates
            spherical_x = torch.cos(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_y = torch.sin(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_z = torch.sin(lats * math.pi / 180) + 1 + 1e-7
            coord_embed_spherical_x = self.coord_expansion(spherical_x, embed_dim//3)
            coord_embed_spherical_y = self.coord_expansion(spherical_y, embed_dim//3)
            coord_embed_spherical_z = self.coord_expansion(spherical_z, embed_dim//3)
            coord_embed = torch.cat([coord_embed_spherical_x, coord_embed_spherical_y, coord_embed_spherical_z], dim=-1) # [B,D]
        elif self.pos_option=='lonlat':
            lons = lons + 180
            lats = lats + 90
            coord_embed_lon = self.coord_expansion(lons, embed_dim//2)
            coord_embed_lat = self.coord_expansion(lats, embed_dim//2)
            coord_embed = torch.cat([coord_embed_lon, coord_embed_lat], dim=-1) # [B,D]

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



    def forward_encoder(self, key, x, mask_ratio, wave_list, bandwidth, input_mode, meta_info, kernel_size=None):

        if isinstance(wave_list, list) and isinstance(bandwidth, list):
            waves = torch.tensor(wave_list, device=x.device).float()
            bandwidths = torch.tensor(bandwidth, device=x.device).float()
        else: # torch tensor
            waves = wave_list.float().to(x.device)
            bandwidths = bandwidth.float().to(x.device)

        if input_mode == 'spectral':
            x,waves = self.patch_embed_spectral(x, waves, bandwidths, kernel_size)
            self.waves = waves
        elif input_mode == 'variable':
            x,waves = self.patch_embed_variable(key, x, waves, bandwidths, kernel_size)
            self.waves = waves

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        pos_embed = resize_abs_pos_embed(self.pos_embed, num_patches_sqrt, (num_patches_sqrt_origin,num_patches_sqrt_origin), num_prefix_tokens=1)

        # coord, scale and time pos embed
        lons, lats, times, areas = meta_info[:, 0], meta_info[:, 1], meta_info[:, 2], meta_info[:, 3]
        embed_dim = pos_embed.shape[-1]
        #pdb.set_trace()
        if torch.isnan(lons).any():
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


        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        if mask_ratio > 0:
            x = self.norm(x) # [B, L-L*ratio, D]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, input_mode, meta_info, kernel_size=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        decoder_pos_embed = resize_abs_pos_embed(self.decoder_pos_embed, num_patches_sqrt, (num_patches_sqrt_origin,num_patches_sqrt_origin), num_prefix_tokens=1)

        # coord, scale and time pos embed
        lons, lats, times, areas = meta_info[:, 0], meta_info[:, 1], meta_info[:, 2], meta_info[:, 3]
        embed_dim = decoder_pos_embed.shape[-1]
        if torch.isnan(lons).any():
            coord_embed = self.coord_token_dec
        else:
            coord_embed = self.get_coord_pos_embed(lons, lats, embed_dim)
        coord_embed = self.coord_fc_decoder(coord_embed)
        if torch.isnan(areas).any():
            area_embed = self.scale_token_dec
        else:
            area_embed = self.get_area_pos_embed(areas, embed_dim)
        area_embed = self.scale_fc_decoder(area_embed)
        if torch.isnan(times).any():
            time_embed = self.time_token_dec
        else:
            time_embed = self.get_time_pos_embed(times, embed_dim)
        time_embed = self.time_fc_decoder(time_embed)
        decoder_pos_embed = decoder_pos_embed + coord_embed + area_embed + time_embed

        # add pos embed
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # DOFA: predictor projection
        #x = self.decoder_pred(x)
        if input_mode == 'spectral':
            x = self.decoder_pred_spectral(x, self.waves, kernel_size)
        elif input_mode == 'variable':
            x = self.decoder_pred_variable(x, self.waves, kernel_size)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, patch_size=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs, patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_single(self, key, imgs, wave_list, bandwidth, meta_info, mask_ratio=0.75, input_mode='spectral', kernel_size=None):
        if key in ['s2_toa_rgb']:
            # no masking for distillation
            latent, _, _ = self.forward_encoder(key, imgs, 0, wave_list, bandwidth, input_mode, meta_info, kernel_size)
            return latent
        latent, mask, ids_restore = self.forward_encoder(key, imgs, mask_ratio, wave_list, bandwidth, input_mode, meta_info, kernel_size)
        pred = self.forward_decoder(latent, ids_restore, input_mode, meta_info, kernel_size)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, kernel_size)
        return loss, pred, mask


    def forward_zt(self, imgs):
        '''
        imgs: [N, 3, H, W] RGB extracted from S2
        '''
        with torch.no_grad():
            assert imgs.size(1) == 3
            #z_t = self.teacher(imgs) # [N, D]
            z_t = self.teacher.get_intermediate_layers(imgs,1)[0]
            z_t = self.teacher_avgpool(z_t.transpose(1, 2))
            z_t = torch.flatten(z_t, 1)

            return z_t


    def forward(self, samples, wvs, bws, metas, drop_prob=0.3, mask_ratio=0.75, kernel_size=None):

        loss_total_mae = 0
        count = 0
        losses = {}
        preds = {}
        masks = {}
        for key in samples.keys():
            img = samples[key]
            wv = wvs[key]
            bw = bws[key]
            meta = metas[key]
            ks = kernel_size[key]
            input_mode = 'spectral' if key in ['s1_grd', 's2_toa', 's3_olci', 's2_toa_rgb'] else 'variable'

            # randomly drop some channel
            if key in ['s2_toa', 's3_olci']:
                img, wv, bw = self.random_select_channels(img, wv, bw)

            # randomly drop some meta 
            if torch.rand(1) < drop_prob:
                meta[:, 0] = torch.tensor(float('nan')).to(meta)
                meta[:, 1] = torch.tensor(float('nan')).to(meta)
            if torch.rand(1) < drop_prob:
                meta[:, 2] = torch.tensor(float('nan')).to(meta)
            if torch.rand(1) < drop_prob:
                meta[:, 3] = torch.tensor(float('nan')).to(meta)

            if key in ['s2_toa_rgb']:
                latent = self.forward_single(key, img, wv, bw, meta, 0, input_mode, ks)
                z_s = self.teacher_avgpool(latent.transpose(1, 2))
                z_s = torch.flatten(z_s, 1)
                z_s = self.student_proj(z_s)
                z_t = self.forward_zt(img)
                loss_distill = (1-(self.cos(z_s, z_t.detach()).mean())) * self.teacher_alpha
                continue
            else:
                loss, pred, mask = self.forward_single(key, img, wv, bw, meta, mask_ratio, input_mode, ks)
                losses[key] = loss
                preds[key] = pred
                masks[key] = mask
                loss_total_mae += loss
                count += 1

        torch.cuda.empty_cache()

        loss_mae = loss_total_mae/count
        total_loss = loss_mae + loss_distill
        
        return total_loss, loss_mae, loss_distill, losses, preds, masks


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
