#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export CUDA_VISIBLE_DEVICES=0
#export GEO_BENCH_DIR=/home/xshadow/Datasets/geobench/
export MODEL_WEIGHTS_DIR=/home/ge86wof2/weights/

model=$1        # Pass the model as the first argument
dataset=$2      # Pass the dataset as the second argument
task=$3         # Pass the task as the third argument
batch_size=$4   # Pass the batch size as the fourth argument
lr=$5           # Pass the learning rate as the fifth argument
epochs=$6       # Pass the number of epochs as the sixth argument
warmup_epochs=$7
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python src/main.py \
output_dir=logs/"${model}_${dataset}" \
model=$model \
dataset=$dataset \
task=$task \
num_gpus=$num_gpus \
num_workers=8 \
batch_size=$batch_size \
epochs=$epochs \
lr=$lr \
warmup_epochs=${warmup_epochs:-0} \
seed=42
