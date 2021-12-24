#!/usr/bin/env bash

gpus=5,6

data_name=Vaihingen
net_G=pspnet
split=test
project_name=MFT_pspnet_Vaihingen_w2_lr0.01_500_poly
checkpoint_name=best_ckpt.pt

python eval.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}

