import torch
import torch.nn as nn
from .CopernicusFM.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .CopernicusFM.models_dwv import vit_large_patch16 as vit_large_patch16_cls
from .CopernicusFM.models_dwv import vit_small_patch16 as vit_small_patch16_cls
from .CopernicusFM.models_dwv_seg import vit_base_patch16 as vit_base_patch16_seg
from .CopernicusFM.models_dwv_seg import vit_large_patch16 as vit_large_patch16_seg
from .CopernicusFM.models_dwv_seg import vit_small_patch16 as vit_small_patch16_seg
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from util.misc import resize
from .lightning_task import LightningTask
from timm.models.layers import trunc_normal_
from util.misc import seg_metric, cls_metric, reg_metric
#from huggingface_hub import hf_hub_download
from torchvision.datasets.utils import download_url
import os

import pdb


class CopernicusFMClassification(LightningTask):

    url = "https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        if model_config.model_size == "base":
            self.encoder = vit_base_patch16_cls(num_classes=data_config.num_classes)
        elif model_config.model_size == "large":
            self.encoder = vit_large_patch16_cls(num_classes=data_config.num_classes)
        elif model_config.model_size == "small":
            self.encoder = vit_small_patch16_cls(num_classes=data_config.num_classes)

        # look for pretrained weights
        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            # hf_hub_download(
            #     repo_id="wangyi111/Copernicus-FM",
            #     filename=filename,
            #     cache_dir=dir,
            #     local_dir=dir,
            # )
            download_url(self.url.format(filename), dir, filename=filename)

        # Load pretrained weights
        check_point = torch.load(path)
        if 'model' in check_point:
            state_dict = check_point['model']
        else:
            state_dict = check_point
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        assert msg.missing_keys==['fc_norm.weight', 'fc_norm.bias'] or msg.missing_keys==['fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias']
        
        # load variable language embedding
        if data_config.language_embed is not None:
            language_path = os.path.join(dir, data_config.language_embed)
            self.language_embed = torch.load(language_path)
            self.language_embed = self.language_embed[data_config.key]
        else:
            self.language_embed = None
        

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        trunc_normal_(self.encoder.head.weight, std=0.01)
        self.encoder.head = nn.Sequential(
            nn.BatchNorm1d(self.encoder.head.in_features, affine=False, eps=1e-6),
            self.encoder.head,
        )
        self.unfreeze(self.encoder.head)

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples, metas):
        out_logits, feats = self.encoder(samples, metas, self.data_config.key, self.data_config.band_wavelengths, self.data_config.band_bandwidths, self.language_embed, self.data_config.input_mode, self.data_config.kernel_size)
        return (out_logits, feats) #if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        return self.encoder.head.parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)


class CopernicusFMSegmentation(LightningTask):

    url = "https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        if model_config.model_size == "base":
            self.encoder = vit_base_patch16_seg()
        elif model_config.model_size == "large":
            self.encoder = vit_large_patch16_seg()
        elif model_config.model_size == "small":
            self.encoder = vit_small_patch16_seg()


        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            # hf_hub_download(
            #     repo_id="wangyi111/Copernicus-FM",
            #     filename=filename,
            #     cache_dir=dir,
            #     local_dir=dir,
            # )
            download_url(self.url.format(filename), dir, filename=filename)

        # Load pretrained weights
        check_point = torch.load(path)
        if 'model' in check_point:
            state_dict = check_point['model']
        else:
            state_dict = check_point
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        assert msg.missing_keys==[]
        
        # load variable language embedding
        if data_config.language_embed is not None:
            language_path = os.path.join(dir, data_config.language_embed)
            self.language_embed = torch.load(language_path)
            self.language_embed = self.language_embed[data_config.key]
        else:
            self.language_embed = None

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=data_config.ignore_index)

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples, metas):
        feats = self.encoder(samples, metas, self.data_config.key, self.data_config.band_wavelengths, self.data_config.band_bandwidths, self.language_embed, self.data_config.input_mode, self.data_config.kernel_size)
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)


class CopernicusFMRegression(LightningTask):

    url = "https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        if model_config.model_size == "base":
            self.encoder = vit_base_patch16_seg()
        elif model_config.model_size == "large":
            self.encoder = vit_large_patch16_seg()
        elif model_config.model_size == "small":
            self.encoder = vit_small_patch16_seg()


        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            # hf_hub_download(
            #     repo_id="wangyi111/Copernicus-FM",
            #     filename=filename,
            #     cache_dir=dir,
            #     local_dir=dir,
            # )
            download_url(self.url.format(filename), dir, filename=filename)

        # Load pretrained weights
        check_point = torch.load(path)
        if 'model' in check_point:
            state_dict = check_point['model']
        else:
            state_dict = check_point
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        assert msg.missing_keys==[]
        
        # load variable language embedding
        if data_config.language_embed is not None:
            language_path = os.path.join(dir, data_config.language_embed)
            self.language_embed = torch.load(language_path)
            self.language_embed = self.language_embed[data_config.key]
        else:
            self.language_embed = None

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            #loss_decode=dict(
            #    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            #),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            # loss_decode=dict(
            #     type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            # ),
        )
        if self.data_config.masknan:
            self.criterion = torch.nn.L1Loss(reduction='none')
        else:
            self.criterion = torch.nn.L1Loss()

    def loss(self, outputs, labels):

        #pdb.set_trace()

        if self.data_config.masknan:
            # qmask not nan
            #qmask = 1-torch.isnan(labels).float()
            #labels_new = labels.clone()
            #labels_new[labels.isnan()] = 0

            loss_pix = self.criterion(outputs[0], labels) + 0.4 * self.criterion(outputs[1], labels)
            #loss_total = (loss_pix * qmask).sum() / qmask.sum()
            loss_total = loss_pix.nanmean()
        else:
            loss_total = self.criterion(outputs[0], labels) + 0.4 * self.criterion(outputs[1], labels)

        return loss_total

    def forward(self, samples, metas):
        feats = self.encoder(samples, metas, self.data_config.key, self.data_config.band_wavelengths, self.data_config.band_bandwidths, self.language_embed, self.data_config.input_mode, self.data_config.kernel_size)
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a
        # return out, out

    def params_to_optimize(self):
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):

        #miou, acc = seg_metric(self.data_config, outputs[0], targets)
        # qmask = 1-torch.isnan(targets).float()
        # targets[targets.isnan()] = 0
        # rmse = masked_root_mean_squared_error(outputs[0], targets, qmask)
        rmse = reg_metric(self.data_config, outputs[0], targets)
        rmse = rmse * self.data_config.target_stats['std'][0]
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)
        #self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)



class CopernicusFMChange(LightningTask):

    url = "https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        if model_config.model_size == "base":
            self.encoder = vit_base_patch16_seg()
        elif model_config.model_size == "large":
            self.encoder = vit_large_patch16_seg()
        elif model_config.model_size == "small":
            self.encoder = vit_small_patch16_seg()


        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            # hf_hub_download(
            #     repo_id="wangyi111/Copernicus-FM",
            #     filename=filename,
            #     cache_dir=dir,
            #     local_dir=dir,
            # )
            download_url(self.url.format(filename), dir, filename=filename)

        # Load pretrained weights
        check_point = torch.load(path)
        if 'model' in check_point:
            state_dict = check_point['model']
        else:
            state_dict = check_point
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        assert msg.missing_keys==[]
        
        # load variable language embedding
        if data_config.language_embed is not None:
            language_path = os.path.join(dir, data_config.language_embed)
            self.language_embed = torch.load(language_path)
            self.language_embed = self.language_embed[data_config.key]
        else:
            self.language_embed = None

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=data_config.ignore_index)

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples, metas):
        B,C,H,W = samples.shape
        samples_pre = samples[:,:C//2,:,:]
        samples_post = samples[:,C//2:,:,:]
        metas_pre = metas[:,0,:]
        metas_post = metas[:,1,:]
        feats_pre = self.encoder(samples_pre, metas_pre, self.data_config.key, self.data_config.band_wavelengths, self.data_config.band_bandwidths, self.language_embed, self.data_config.input_mode, self.data_config.kernel_size)
        feats_post = self.encoder(samples_post, metas_post, self.data_config.key, self.data_config.band_wavelengths, self.data_config.band_bandwidths, self.language_embed, self.data_config.input_mode, self.data_config.kernel_size)
        
        feats = []
        for i in range(len(feats_pre)):
            feats.append(feats_post[i] - feats_pre[i])
        
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)




# Model factory for different dinov2 tasks
def CopernicusFMModel(args, model_config, data_config):
    if args.task == "classification":
        return CopernicusFMClassification(args, model_config, data_config)
    elif args.task == "segmentation":
        return CopernicusFMSegmentation(args, model_config, data_config)
    elif args.task == "regression":
        return CopernicusFMRegression(args, model_config, data_config)
    elif args.task == "changedetection":
        return CopernicusFMChange(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
