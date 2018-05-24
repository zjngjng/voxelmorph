#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model vm1 \
    --save_name exp_resized_lambda1.0\
    --gpu 0 \
    --lambda 1\
    --lr 0.0001 \
    --checkpoint_iter 5000
