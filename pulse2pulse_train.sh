
#!/bin/bash

clear && python pulse2pulse_train.py train \
    --exp_name pulse2pulse_model \
    --device_id 1 \
    --num_epochs 1335 \
    --bs 32 \
    --checkpoint_interval 10 \
    --save_interval 100 \
    --fold_idx 0 \

