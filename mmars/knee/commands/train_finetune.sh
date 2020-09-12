#!/bin/bash

my_dir="$(dirname "$0")"
#. $my_dir/set_env.sh
export MMAR_ROOT=/var/nvidia/aiaa/mmars/knee/

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
CONFIG_FILE=config/config_train.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/dataset_aiaa.json \
    epochs=2000 \
    learning_rate=0.000005 \
    num_training_epoch_per_valid=20 \
    MMAR_CKPT=$MMAR_ROOT/models/model.ckpt \
    multi_gpu=false
