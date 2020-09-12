#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
CONFIG_FILE=config/config_train.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    --set \
    DATA_ROOT=/workspace/mmars/covid19/trainingdataio_covid19_ct_lung_seg_v1/data \
    DATASET_JSON=/workspace/mmars/covid19/trainingdataio_covid19_ct_lung_seg_v1/config/dataset_0.json \
    epochs=1200 \
    learning_rate=0.0001 \
    num_training_epoch_per_valid=20 \
    multi_gpu=false
