#!/bin/bash

clear && python wavegan_star_train.py train \
    --exp_name wavegan_star_model \
    --checkpoint_interval 10 \
    --save_interval 100 \
    --num_epochs 500 \
    --start_epoch 0 \
    --bs 32 \
    --lr 0.0001 \
    --b1 0.5 \
    --b2 0.9 \
    --device_id 2 \
    --fold_idx 4 \
    --checkpoint_path /gim/lv02/isviridov/code/gans/gan_ecg/gan_models/Pulse2Pulse/output/wavegan_star_model/cps/gan_p2p_epoch:300.pt