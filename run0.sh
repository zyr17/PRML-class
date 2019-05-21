#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
echo $(date "+%Y-%m-%d %H:%M:%S") >> log.txt
nohup python -u main.py $* >> log.txt 2>&1 &
