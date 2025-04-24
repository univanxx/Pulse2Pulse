
#!/bin/bash

export DATA_DIR="your_dir"

clear && python pulse2pulse_train.py \
    --action train \
    --data_dir $DATA_DIR \
    --exp_name pulse2pulse_model \
    --device_id 1 \
    --num_epochs 2400 \
    --bs 32 \
    --checkpoint_interval 100 \
    --save_interval 100