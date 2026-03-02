#!/bin/bash
uv run python src/main.py train \
    --epochs 20 \
    --bs 256 \
    --eval_metric r@10 \
    --alpha 0.8 \
    --lambda_neg 0.8 \
    --gamma 0.5 \
    --eval_start_epoch 0 \
    --eval_freq 1 \
    --seed 42