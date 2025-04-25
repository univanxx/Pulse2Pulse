#!/bin/bash

python generate_signals.py --checkpoint "your_checkpoint_path" \
    --labels_path "your_labels_path" \
    --task_type "addition" \
    --save_path "your_save_path" \
    --batch_size 32 \
    --device_id "your_device_id" \
    --model_name "wg*"