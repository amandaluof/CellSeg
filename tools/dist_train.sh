#!/usr/bin/env bash
###
 # @Author: your name
 # @Date: 2021-10-21 21:45:12
 # @LastEditTime: 2021-10-27 20:53:03
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: /lf/PolarMask/tools/dist_train.sh
### 

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 69512\
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
