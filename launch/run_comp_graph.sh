#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
nohup python -u scripts/comp_graphs/train.py --dispatch --start 0 --end 1 > logs/train_full.txt &
sleep 3

export CUDA_VISIBLE_DEVICES=1
nohup python -u scripts/comp_graphs/train.py --dispatch --start 1 --end 2 > logs/train_spix.txt &
sleep 3

export CUDA_VISIBLE_DEVICES=2
nohup python -u scripts/comp_graphs/train.py --dispatch --start 2 --end 3 > logs/train_sprobs.txt &
sleep 3
