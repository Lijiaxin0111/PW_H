@echo off

set NUM_GPUS_PER_WORKER=1
set MASTER_PORT=6008

set train_options=--savepath blip_uni_cross_mul ^
--batch_size 16 ^
--accumulation_steps 4 ^
--epochs 10 ^
--distributed False ^
--gpu_num %NUM_GPUS_PER_WORKER% ^
--gpu_id '0,1,2,3' ^
--clear_visualizer ^
--fix_rate 0.7 ^
--lr 1e-05 ^
--lr-decay-style cosine ^
--split train
--warmup 0.0 ^

--valid_per_epoch 4 ^
--dataroot "/home/jiaxin/Composed_BLIP/IQA_data/AIGCIQA2023"


set "run_cmd=  python  -u  ./train.py %train_options%"

echo %run_cmd%
%run_cmd%