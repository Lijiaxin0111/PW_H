#!/bin/bash

NUM_GPUS_PER_WORKER=0
MASTER_PORT=6008
CUDA_VISIBLE_DEVICES=0

train_options="--savepath blip_uni_cross_mul_more_epoch \
--expri test_cross_attention_exp_lr_10_5 \
--batch_size 16 \
--dataset whole \
--mod Mix_TXT \
--accumulation_steps 2 \
--epochs 20 \
--gpu_num $NUM_GPUS_PER_WORKER \
--fix_rate 0.5 \
--lr 1e-5 \
--lr-decay-style cosine \
--lr-decay-ratio 0 \
--split train \
--warmup 0.0 \
--valid_per_epoch 4  \
--preload_path /mnt/lustre/sjtu/home/yzl02/Composed_BLIP/PW_H/checkpoint/test_aus_MLNET_exp_lr_10_3_blip_uni_cross_mul_more_epoch_bs0_fix=0.0_lr=0.001/current_lr=0.001.pt "


run_cmd="  python  ./train.py $train_options"

echo $run_cmd
$run_cmd