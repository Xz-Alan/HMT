import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.modules import transformer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, utils

import datasets.custom_transforms as ct
from datasets.Seg_dataset import SegDataset


def get_loader(root_dir, data_str, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='SegDataset'):
    if not osp.isdir(osp.join(root_dir, 'list')):
        input("no list")
        generate_list(root_dir)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_transform = transforms.Compose([# ct.RandomSized(256),
                                          # ct.RandomRotate(15),
                                          # ct.RandomHorizontalFlip(),
                                          ct.Normalize(mean=mean, std=std),
                                          ct.ToTensor()])
    if dataset == 'SegDataset':
        data_set = SegDataset(root_dir=root_dir, split=split, data_str=data_str, img_size=img_size, is_train=is_train, transform=test_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [SegDataset,])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=2)

    return dataloader


def get_loaders(args):
    if not osp.isdir(osp.join(args.root_dir, 'list')):
        input("no list")
        generate_list(args.root_dir)
    split = args.split
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # TODO: img & label transform
    train_transform = transforms.Compose([#ct.RandomSized(256),
                                          #ct.RandomRotate(15),
                                          ct.RandomHorizontalFlip(),
                                          ct.Normalize(mean=mean, std=std),
                                          ct.ToTensor()])
    valid_transform = transforms.Compose([# ct.RandomSized(256),
                                          # ct.RandomRotate(15),
                                          # ct.RandomHorizontalFlip(),
                                          ct.Normalize(mean=mean, std=std),
                                          ct.ToTensor()])
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'SegDataset':
        training_set = SegDataset(root_dir=args.root_dir, split=split, data_str=args.data_str, img_size=args.img_size, is_train=True, transform=train_transform)
        val_set = SegDataset(root_dir=args.root_dir, split=split_val, data_str=args.data_str, img_size=args.img_size, is_train=False, transform=valid_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [SegDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    batch_sizes = {'train': args.batch_size, 'val': 1}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_sizes[x],
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders

def generate_list(root_dir):
    dtypes = ['train', 'valid', 'test']
    ctypes = ['gts']
    os.mkdir(root_dir + 'list')
    for dtype in dtypes:
        for ctype in ctypes:
            f = open(osp.join(root_dir, 'list', dtype + '.txt'), 'w')
            for filename in os.listdir(osp.join(root_dir, ctype, dtype)):
                filename = filename.split('.')[0]
                f.write(filename + '\n')
    return 1


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

        