#!/usr/bin/env bash

gpus=4,5,6,7
checkpoint_root=checkpoints
data_name=Vaihingen       # Potsdam | Vaihingen | GID

num_workers=2
img_size=512
batch_size=8
lr=0.01
max_epochs=1000
net_G=wetr
#base_resnet18
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_dedim8
lr_policy=poly    # step | linear | poly

project_name=MFT_${net_G}_${data_name}_w${num_workers}_lr${lr}_${max_epochs}_${lr_policy}_s${img_size}_mit_b5

python train.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --num_workers ${num_workers} 
# --local_rank 6
