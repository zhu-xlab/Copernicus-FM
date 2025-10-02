#!/bin/bash
# CPU debugging version with minimal resources
export CUDA_VISIBLE_DEVICES=""

python main_pretrain.py \
--data_mode webdataset \
--trainshards data/copernicus-pretrain-tar/example-{000000..000001}.tar \
--dataset_size 10 \
--shuffle 5 \
--output_dir ./checkpoints_cpu_debug \
--log_dir ./checkpoints_cpu_debug/log \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.7 \
--num_workers 1 \
--batch_size 1 \
--epochs 1 \
--warmup_epochs 0 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--distill_size base \
--device cpu \
--accum_iter 1