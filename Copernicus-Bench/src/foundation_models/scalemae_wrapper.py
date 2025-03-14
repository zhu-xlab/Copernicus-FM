from .ScaleMAE.models_vit_seg import vit_large_patch16 as vit_large_patch16_seg
from .ScaleMAE.models_vit import vit_large_patch16 as vit_large_patch16_cls

import torch.nn as nn
import torch
import os

# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from util.misc import resize
from .lightning_task import LightningTask
from util.misc import seg_metric, cls_metric
from torchvision.datasets.utils import download_url


class ScaleMAEClassification(LightningTask):

    url = "https://huggingface.co/torchgeo/{}/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = vit_large_patch16_cls(
            img_size=model_config.image_resolution, in_chans=3, drop_path_rate=0.0
        )

        # Load pretrained weights
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
        # logger.debug(msg)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)  # look for pretrained weights
        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            download_url(self.url.format(filename.split(".")[0], filename), dir, filename=filename)

        self.linear_classifier = torch.nn.Linear(
            model_config.embed_dim, data_config.num_classes
        )

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        feats = self.encoder.forward_features(samples)
        out_logits = self.linear_classifier(feats)
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        return self.linear_classifier.parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)


class ScaleMAESegmentation(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.encoder = vit_large_patch16_seg(
            img_size=model_config.image_resolution, in_chans=3, drop_path_rate=0.0
        )

        # Load pretrained weights
        checkpoint = torch.load(model_config.pretrained_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)

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
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples):
        feats = self.encoder(samples)
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
def ScaleMAEModel(args, model_config, data_config):
    if args.task == "classification":
        return ScaleMAEClassification(args, model_config, data_config)
    elif args.task == "segmentation":
        return ScaleMAESegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
