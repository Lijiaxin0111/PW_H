#!/bin/bash

NUM_GPUS_PER_WORKER=0
MASTER_PORT=6008
CUDA_VISIBLE_DEVICES=0

train_options="--savepath blip_uni_cross_mul_more_epoch \
--expri test_image \
--batch_size 16 \
--dataset 3 \
--mod MIX \
--accumulation_steps 2 \
--epochs 20 \
--gpu_num $NUM_GPUS_PER_WORKER \
--fix_rate 0 \
--lr 1e-4 \
--lr-decay-style cosine \
--lr-decay-ratio 0 \
--split train \
--warmup 0.0 \
--valid_per_epoch 4  \
--preload_path /mnt/homes/jiaxin/PW_H/checkpoint/Train_in_3_after_all_MIX_exp_lr_10_4_blip_uni_cross_mul_more_epoch_bs0_fix=0.0_lr=0.0001/best_lr=0.0001.pt"


run_cmd="  python  ./train.py $train_options"

echo $run_cmd
$run_cmd