# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate
from huggingface_hub import hf_hub_download

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from loguru import logger


def load_checkpoint(config, model, optimizer, lr_scheduler):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if (
            "amp" in checkpoint
            and config.AMP_OPT_LEVEL != "O0"
            and checkpoint["config"].AMP_OPT_LEVEL != "O0"
        ):
            amp.load_state_dict(checkpoint["amp"])
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }
    if config.AMP_OPT_LEVEL != "O0":
        save_state["amp"] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, model):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    # look for pretrained weights
    dir = os.getenv("MODEL_WEIGHTS_DIR")
    filename = model_config.pretrained_path
    path = os.path.join(dir, filename)
    if not os.path.exists(path):
        # download the weights from HF
        hf_hub_download(
            repo_id="earthflow/GeoFMs", filename=filename, cache_dir=dir, local_dir=dir
        )
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    if any([True if "encoder." in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint_model.items()
            if k.startswith("encoder.")
        }
        logger.info("Detect pre-trained model, remove [encoder.] prefix.")
    else:
        logger.info("Detect non-pre-trained model, pass without doing anything.")

    if config.MODEL.TYPE == "swin":
        logger.info(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == "vit":
        logger.info(">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    else:
        raise NotImplementedError
    # MATIAS: If using ImageNet (3 channel) pretrained weights for Sentinel-2 (12 band) data
    if (
        model.patch_embed.proj.weight.shape
        != checkpoint_model["patch_embed.proj.weight"].shape
    ):
        temp = model.patch_embed.proj.weight.data.cpu()
        if checkpoint_model["patch_embed.proj.weight"].shape[1] == 1:
            # greyscale pretrained model
            temp = checkpoint_model["patch_embed.proj.weight"].repeat(
                1, temp.shape[1], 1, 1
            )
        elif (
            checkpoint_model["patch_embed.proj.weight"].shape[1] == 12
            and temp.shape[1] == 3
        ):
            # For 12 band pretrained, the order is CGBR...
            temp[:, :, :, :] = checkpoint_model["patch_embed.proj.weight"][
                :, [3, 2, 1], :, :
            ]
        elif checkpoint_model["patch_embed.proj.weight"].shape[1] == 8:
            # SpaceNet superres pretrain
            min_channels = min(
                temp.shape[1], checkpoint_model["patch_embed.proj.weight"].shape[1]
            )
            temp[:, :min_channels, :, :] = checkpoint_model["patch_embed.proj.weight"][
                :, :min_channels, :, :
            ]
        else:
            temp[:, [3, 2, 1], :, :] = checkpoint_model["patch_embed.proj.weight"]
        checkpoint_model["patch_embed.proj.weight"] = temp
    if "finetune" in config.PRETRAINED and hasattr(model, "head"):
        logger.info(
            f">>>>>>>>>> Resizing head from supervised model {config.PRETRAINED} .........."
        )
        temp_headw = model.head.weight.data.cpu()
        temp_headb = model.head.bias.data.cpu()
        checkpoint_model["head.weight"] = temp_headw
        checkpoint_model["head.bias"] = temp_headb
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.3' not in k}
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.2.blocks.5' not in k}
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.2.blocks.4' not in k}
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # if 'finetune' in config.PRETRAINED:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.head.parameters():
    #         param.requires_grad = True
    #     for n, param in model.named_parameters():
    #         layers = ['layers.0.blocks.0.attn.relative_position_index', 'layers.0.blocks.1.attn_mask', 'layers.0.blocks.1.attn.relative_position_index', 'layers.1.blocks.0.attn.relative_position_index',
    #             'layers.1.blocks.1.attn_mask', 'layers.1.blocks.1.attn.relative_position_index', 'layers.2.blocks.0.attn.relative_position_index', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.1.attn.relative_position_index',
    #              'layers.2.blocks.2.attn.relative_position_index', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.3.attn.relative_position_index', 'layers.2.blocks.4.attn.relative_position_index', 'layers.2.blocks.5.attn_mask']
    #         if 'layer3' in n or n in layers:
    #             param.requires_grad = True
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")


def load_continual(config, model):
    logger.info(f">>>>>>>>>> Loaded from {config.PRETRAINED} ..........")
    checkpoint = torch.load(config.PRETRAINED, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    # if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
    #     checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
    #     logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    # else:
    #     logger.info('Detect non-pre-trained model, pass without doing anything.')
    checkpoint_model = {"encoder.{}".format(k): v for k, v in checkpoint_model.items()}
    if config.MODEL.TYPE == "swin":
        logger.info(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == "vit":
        logger.info(">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    else:
        raise NotImplementedError
    # print(checkpoint_model)
    # MATIAS: If using ImageNet (3 channel) pretrained weights for Sentinel-2 (12 band) data
    # if model.encoder.patch_embed.proj.weight.shape != checkpoint_model['encoder.patch_embed.proj.weight'].shape:
    #     logger.info(f">>>>>>>>>> Reshaping... '{config.PRETRAINED}'")
    #     temp = model.encoder.patch_embed.proj.weight.data.cpu()
    #     temp[:,[3,2,1],:,:] = checkpoint_model['encoder.patch_embed.proj.weight']
    #     checkpoint_model['encoder.patch_embed.proj.weight'] = temp

    #     temp = model.decoder[0].weight.data.cpu()
    #     temp2 = model.decoder[0].bias.data.cpu()
    #     temp[3072:4096,:,:,:] = checkpoint_model['decoder.0.weight'][0:1024,:,:,:] #3
    #     temp[2048:3072,:,:,:] = checkpoint_model['decoder.0.weight'][1024:2048,:,:,:] #2
    #     temp[1024:2048,:,:,:] = checkpoint_model['decoder.0.weight'][2048:3072,:,:,:] #1
    #     temp2[3072:4096] = checkpoint_model['decoder.0.bias'][0:1024] #3
    #     temp2[2048:3072] = checkpoint_model['decoder.0.bias'][1024:2048] #2
    #     temp2[1024:2048] = checkpoint_model['decoder.0.bias'][2048:3072] #1
    #     checkpoint_model['decoder.0.weight'] = temp
    #     checkpoint_model['decoder.0.bias'] = temp2
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()
    print(state_dict)
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            print(key)
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(
                        f"{key}: Interpolate relative_position_bias_table using geo."
                    )
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = (
                            relative_position_bias_table_pretrained[:, i]
                            .view(src_size, src_size)
                            .float()
                            .numpy()
                        )
                        f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(relative_position_bias_table_pretrained.device)
                        )

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in checkpoint_model.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [
        k for k in checkpoint_model.keys() if "relative_coords_table" in k
    ]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model):
    # Duplicate shared rel_pos_bias to each layer
    if (
        getattr(model, "use_rel_pos_bias", False)
        and "rel_pos_bias.relative_position_bias_table" in checkpoint_model
    ):
        logger.info(
            "Expand the shared relative position embedding to each transformer block."
        )
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = (
                rel_pos_bias.clone()
            )
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info(
                    "Position interpolate for %s from %dx%d to %dx%d"
                    % (key, src_size, src_size, dst_size, dst_size)
                )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model
