#!/usr/bin/env bash

gpus=5,6

data_name=Vaihingen
net_G=wetr
split=test
project_name=MFT_wetr_Vaihingen_w2_lr0.01_1000_poly_mit_b5
checkpoint_name=best_ckpt.pt

python eval.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}

