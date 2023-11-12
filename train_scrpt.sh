#!/bin/bash

NUM_GPUS_PER_WORKER=0
MASTER_PORT=6008

train_options="--savepath blip_uni_cross_mul_more_epoch \
--expri test_3k_aug_yes_com_yes_epoch_10 \
--preload_path /home/jiaxin/Composed_BLIP/checkpoint/ImageReward.pt \
--batch_size 2 \
--dataset AGIQA_3K \
--accumulation_steps 2 \
--epochs 10 \
--gpu_num $NUM_GPUS_PER_WORKER \
--fix_rate 0.3 \
--lr 1e-5 \
--lr-decay-style cosine \
--lr-decay-ratio 0 \
--split train \
--warmup 0.0 \
--valid_per_epoch 4 \
--stage 2 "

run_cmd="python -u ./train.py $train_options"

echo $run_cmd
$run_cmd