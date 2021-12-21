import functools

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler

import models


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = models.UpResNet(input_nc=3, output_nc=args.n_class, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = models.BASE_Transformer(input_nc=3, output_nc=args.n_class, token_len=4, resnet_stages_num=4, with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = models.BASE_Transformer(input_nc=3, output_nc=args.n_class, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = models.BASE_Transformer(input_nc=3, output_nc=args.n_class, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    elif args.net_G == 'pspnet':
        net = models.PSPNet(n_classes=args.n_class, backend='resnet101')
    elif args.net_G == 'segformer':
        net = models.Segformer(num_classes=args.n_class)
    elif args.net_G == 'segmenter':
        net = models.create_segmenter(args.n_class, patch_size=8, decoder_name='mask')
    elif args.net_G == 'wetr':
        net = models.WeTr(backbone='mit_b5', num_classes=args.n_class, embedding_dim=256, pretrained=False)  # mit_b0--b5
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, gpu_ids)
        # net = DistributedDataParallel(net, gpu_ids)
    # init_weights(net, init_type==init_type, init_gain=init_gain)    # normal | xavier | kaiming | orthogonal
    return net


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - epoch / float(args.max_epochs + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'poly':
        def lambda_poly(epoch):
            return (1 - float(epoch) / args.max_epochs) ** 0.9
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_weights_2(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


