#!/bin/bash
# GPU debugging version with minimal resources
torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=29501 main_pretrain.py \
--data_mode webdataset \
--trainshards data/copernicus-pretrain-tar/example-{000000..000004}.tar \
--dataset_size 500 \
--shuffle 20 \
--output_dir ./checkpoints \
--log_dir ./checkpoints/log \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.7 \
--num_workers 2 \
--batch_size 2 \
--epochs 2 \
--warmup_epochs 1 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--distill_size base \
# --dist_url $dist_url \
# --dist_backend 'nccl' \
