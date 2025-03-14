# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from typing import Iterable

import torch
# import wandb

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        images, images_up_2x, images_up_4x = (
            samples["img"],
            samples["img_up_2x"],
            samples["img_up_4x"],
        )
        # images, images_up_2x = samples['img'], samples['img_up_2x']  # for fmow_rgb at 2 scales

        images = images.to(device, non_blocking=True)
        images_up_2x = images_up_2x.to(device, non_blocking=True)
        images_up_4x = images_up_4x.to(
            device, non_blocking=True
        )  # comment for fmow_rgb at 2 scales

        with torch.cuda.amp.autocast():
            mse_loss, l1_loss, _, _ = model(
                images, [images_up_2x, images_up_4x], mask_ratio=args.mask_ratio
            )
            # mse_loss, l1_loss, _, _ = model(images, images_up_2x, mask_ratio=args.mask_ratio) # for fmow_rgb at 2 scales
            loss = 0.6 * mse_loss + 0.4 * l1_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")
            # sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

            """
            # Wandb logging
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass
            """

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
