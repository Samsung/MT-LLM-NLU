#!/bin/bash

port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config.yaml \
    train.py \
    --config configs/bigtranslate_config.yaml