# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Yi Wang.
# All rights reserved.

# This code is adapted from https://github.com/facebookresearch/mae, which is 
# licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file except in compliance with the License.
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import time

# wavelengths (in nm)
WV = {
    "s1_grd": [50000000, 50000000],
    "s2_toa": [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190],
    "s3_olci": [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 
                708.75, 753.75, 761.25, 764.375, 767.5, 778.75, 865, 885, 
                900, 940, 1029],
    "s5p_co": [2325],
    "s5p_no2": [430],
    "s5p_o3": [300],
    "s5p_so2": [320],
    "dem": [0]
}
# bandwidths (in nm)
BW = {
    "s1_grd": [1e9, 1e9],
    "s2_toa": [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    "s3_olci": [15, 10, 10, 10, 10, 10, 10, 10, 7.5, 7.5, 10, 7.5, 7.5, 3.75, 
                2.5, 15, 20, 10, 10, 20, 40],
    "s5p_co": [80],
    "s5p_no2": [60],
    "s5p_o3": [30],
    "s5p_so2": [15],
    "dem": [0]
}
# input sizes
IN_SIZE = {
    "s1_grd": 224,
    "s2_toa": 224,
    "s3_olci": 96,
    "s5p_co": 28,
    "s5p_no2": 28,
    "s5p_o3": 28,
    "s5p_so2": 28,
    "dem": 960    
}
# vit patch sizes / patch embedding kernel sizes
KERNEL_SIZE = {
    "s1_grd": 16,
    "s2_toa": 16,
    "s3_olci": 8,
    "s5p_co": 4,
    "s5p_no2": 4,
    "s5p_o3": 4,
    "s5p_so2": 4,
    "dem": 64
}
# ground sampling distances (in meters)
GSD = {
    "s1_grd": 10,
    "s2_toa": 10,
    "s3_olci": 300,
    "s5p_co": 1000,
    "s5p_no2": 1000,
    "s5p_o3": 1000,
    "s5p_so2": 1000,
    "dem": 30
}
# surface areas (in square km) covered by each small patch (not the whole image)
AREA = {
    "s1_grd": (KERNEL_SIZE["s1_grd"] * GSD["s1_grd"] / 1000) ** 2,
    "s2_toa": (KERNEL_SIZE["s2_toa"] * GSD["s2_toa"] / 1000) ** 2,
    "s3_olci": (KERNEL_SIZE["s3_olci"] * GSD["s3_olci"] / 1000) ** 2,
    "s5p": (KERNEL_SIZE["s5p_no2"] * GSD["s5p_no2"] / 1000) ** 2,
    "dem": (KERNEL_SIZE["dem"] * GSD["dem"] / 1000) ** 2
}
# ground sampling distances (approx. in degrees, used to adjust lon/lat when cropping)
GSD_DEG = {key: value * 0.00009 for key, value in GSD.items()}


def calculate_scale_and_center_shift(crop_coords, original_size):
    """
    Calculate scale factors and center coordinate shifts for cropped images.

    Args:
        crop_coords (torch.Tensor): Tensor of shape (batch_size, 4, 2) containing
                                     the coordinates of the 4 corners of the crop
                                     (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
        original_size (tuple): Tuple (original_width, original_height) of the original image size.

    Returns:
        tuple: A tuple containing:
            - scale_factors (torch.Tensor): Tensor of shape (batch_size,) with area scaling factors.
            - center_shifts (torch.Tensor): Tensor of shape (batch_size, 2) with shifts of crop centers
                                            relative to the original image center.
    """
    original_width, original_height = original_size
    original_area = original_width * original_height
    original_center = torch.tensor([original_width / 2, original_height / 2])

    # Calculate width and height for each crop
    crop_widths = crop_coords[:, 1, 0] - crop_coords[:, 0, 0]
    crop_heights = crop_coords[:, 2, 1] - crop_coords[:, 0, 1]

    # Calculate area for each crop
    crop_areas = crop_widths * crop_heights

    # Compute scaling factors (area ratio)
    scale_factors = crop_areas / original_area

    # Calculate the center of each crop
    crop_centers = torch.stack([
        (crop_coords[:, 0, 0] + crop_coords[:, 1, 0]) / 2,  # x-center
        (crop_coords[:, 0, 1] + crop_coords[:, 2, 1]) / 2   # y-center
    ], dim=1)

    # Calculate the shift of the crop center relative to the original center
    center_shifts = crop_centers - original_center

    return scale_factors, center_shifts

def prepare_metadata(meta, device):
    """Prepare and format metadata for all sensor types."""
    # Pad zero to the last dimension
    meta = torch.cat([meta, torch.zeros(meta.size(0), meta.size(1), 1).to(meta)], dim=2)
    # Assign area to metadata
    meta[:, 0, 3] = AREA["s1_grd"]
    meta[:, 1, 3] = AREA["s2_toa"]
    meta[:, 2, 3] = AREA["s3_olci"]
    meta[:, 3:7, 3] = AREA["s5p"]
    meta[:, 7, 3] = AREA["dem"]

    sample_meta = {
        "s1_grd": meta[:, 0, :].to(device, non_blocking=True),
        "s2_toa": meta[:, 1, :].to(device, non_blocking=True),
        "s3_olci": meta[:, 2, :].to(device, non_blocking=True),
        "s5p_co": meta[:, 3, :].to(device, non_blocking=True),
        "s5p_no2": meta[:, 4, :].to(device, non_blocking=True),
        "s5p_o3": meta[:, 5, :].to(device, non_blocking=True),
        "s5p_so2": meta[:, 6, :].to(device, non_blocking=True),
        "dem": meta[:, 7, :].to(device, non_blocking=True)
    }
    
    return sample_meta


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, train_transforms=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        sample_s1, sample_s2, sample_s3, sample_co, sample_no2, sample_o3, sample_so2, sample_dem, meta = samples
        # meta is B,8,3: lon, lat, time
        sample_meta = prepare_metadata(meta, device)

        t0 = time.time()

        samples_new = {
            "s1_grd": train_transforms['s1_grd'](sample_s1.to(device, non_blocking=True)),
            "s2_toa": train_transforms['s2_toa'](sample_s2.to(torch.float32).to(device, non_blocking=True)),
            "s3_olci": train_transforms['s3_olci'](sample_s3.to(device, non_blocking=True)),
            "s5p_co": train_transforms['s5p_co'](sample_co.to(device, non_blocking=True)),
            "s5p_no2": train_transforms['s5p_no2'](sample_no2.to(device, non_blocking=True)),
            "s5p_o3": train_transforms['s5p_o3'](sample_o3.to(device, non_blocking=True)),
            "s5p_so2": train_transforms['s5p_so2'](sample_so2.to(device, non_blocking=True)),
            "dem": train_transforms['dem'](sample_dem.to(device, non_blocking=True))
        }

        if args.scale_option=='augarea':
            # adjust metadata according to the actual cropped region
            for key in samples_new.keys():
                crop_bbx = train_transforms[key]._params[0].data['src']
                original_size = train_transforms[key]._params[0].data['input_size'][0]
                scale_ratio, center_shifts = calculate_scale_and_center_shift(crop_bbx,original_size)
                scale_ratio = scale_ratio.to(device)
                center_shifts = center_shifts.to(device)
                # adjust area
                sample_meta[key][:,3] = sample_meta[key][:,3] * scale_ratio
                # adjust center coord
                sample_meta[key][:,0] = sample_meta[key][:,0] + center_shifts[:,0] * GSD_DEG[key]
                sample_meta[key][:,1] = sample_meta[key][:,1] - center_shifts[:,1] * GSD_DEG[key]

        if args.distill_size is not None:
            # distill RGB from DINOv2
            samples_new["s2_toa_rgb"] = samples_new["s2_toa"][:, [3,2,1], :, :] # Bx3x224x224
            WV["s2_toa_rgb"] = WV["s2_toa"][1:4][::-1] # r,g,b
            BW["s2_toa_rgb"] = BW["s2_toa"][1:4][::-1]
            KERNEL_SIZE["s2_toa_rgb"] = KERNEL_SIZE["s2_toa"]
            sample_meta["s2_toa_rgb"] = sample_meta["s2_toa"]

        t1 = time.time()
        # print("Data transfer time: ", t1 - t0)
        
        # Use autocast only for CUDA devices
        if device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=False):
                loss, loss_mae, loss_distill, losses_mae, preds, masks = model(samples_new, WV, BW, sample_meta, drop_prob=args.drop_prob, mask_ratio=args.mask_ratio, kernel_size=KERNEL_SIZE)
        else:
            loss, loss_mae, loss_distill, losses_mae, preds, masks = model(samples_new, WV, BW, sample_meta, drop_prob=args.drop_prob, mask_ratio=args.mask_ratio, kernel_size=KERNEL_SIZE)

        t2 = time.time()
        # print("Model time: ", t2 - t1)

        loss_value = loss.item()
        loss_mae_value = loss_mae.item()
        if args.distill_size is not None:
            loss_distill_value = loss_distill.item()
        else:
            loss_distill_value = loss_distill

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        t3 = time.time()
        #print("Backward time: ", t3 - t2)

        # Only synchronize CUDA if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        #torch.cuda.empty_cache()

        t4 = time.time()
        #print("Sync time: ", t4 - t3)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mae=loss_mae_value)
        metric_logger.update(loss_distill=loss_distill_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        t5 = time.time()
        #print("Log time: ", t5 - t4)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



