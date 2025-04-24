#!/bin/bash

export DATA_DIR="your_dir"

clear && python wavegan_star_train.py \
    --action train \
    --data_dir $DATA_DIR \
    --exp_name wavegan_star_model \
    --checkpoint_interval 100 \
    --save_interval 100 \
    --num_epochs 3000 \
    --start_epoch 0 \
    --bs 32 \
    --lr 0.0001 \
    --b1 0.5 \
    --b2 0.9 \
    --device_id 1