import argparse
import json
import logging
import os
import os.path as osp
import pdb
from argparse import ArgumentParser

import cv2
import numpy as np
from IPython import embed
from tqdm import tqdm

from data_config import DataConfig

def label2vis(label, palette_file):
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
    list_value = np.asarray(list(text.values()), dtype=np.uint8)
    vis_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            vis_label[i,j,:] = list_value[label[i,j][0]]
    return vis_label


def pad_concat(args):
    for ori_name in os.listdir(args.init_path):
        print(ori_name)
        ori_gt = cv2.imread(osp.join(args.init_path, ori_name))
        h, w = ori_gt.shape[0], ori_gt.shape[1]
        h_pad = h // args.img_size if (h % args.img_size == 0) else (h // args.img_size + 1)
        w_pad = w // args.img_size if (w % args.img_size == 0) else (w // args.img_size + 1)
        total_pad = h_pad * w_pad
        cont_pred = np.zeros((h_pad * args.img_size, w_pad * args.img_size, ori_gt.shape[2]), dtype=np.uint8)
        for i in range(total_pad):
            row = i // w_pad
            column = i % w_pad
            temp_pred = cv2.imread(osp.join(args.vis_path, ori_name[16:-4] + '_%03d.png'%(i+1)))
            # temp_pred = label2vis(temp_pred, args.palette_path)
            assert args.img_size == temp_pred.shape[0] == temp_pred.shape[1]
            # print(i)
            # pdb.set_trace()
            cont_pred[row*args.img_size:(row+1)*args.img_size, column*args.img_size:(column+1)*args.img_size, :] = temp_pred
        print('cont', cont_pred.shape)
        cont_pred = cont_pred[:h, :w, :]
        print('cont_pred', cont_pred.shape)
        cv2.imwrite(osp.join(args.final_path, ori_name[16:-4] + '.png'), cont_pred)
        input()


def main():
    parser = ArgumentParser()
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--data_name', default='Vaihingen', type=str, help='Potsdam | Vaihingen | GID')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--root_dir', default='', type=str)
    parser.add_argument('--palette_path', default='../data/palette.json', type=str)
    parser.add_argument('--split', default='test', type=str)

    args = parser.parse_args()
    args.project_name = input('project_name: ')
    dataConfig = DataConfig().get_data_config(args.data_name)
    args.root_dir = dataConfig.root_dir

    args.vis_path = os.path.join('vis', args.project_name)
    args.final_path = os.path.join('vis_final', args.data_name, args.project_name)
    os.makedirs(args.final_path, exist_ok=True)
    args.init_path = osp.join(args.root_dir, 'gts_init', args.split)

    pad_concat(args)


if __name__ == "__main__":
    main()




