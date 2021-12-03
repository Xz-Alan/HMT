import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils
from misc.logger_tool import Logger, Timer
from misc.metric_tool import ConfuseMatrixMeter, MetricScore
from numpy.ma.core import argsort, set_fill_value
from PIL import Image
from utils import de_norm

import models.losses as losses
from models.init_networks import *
from models.losses import cross_entropy


class SegTrainer():

    def __init__(self, args, dataloaders):
        self.dataloaders = dataloaders
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_name = args.net_G
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0 else "cpu")
        print(self.device)

        # define optimizers
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)     # 5e-4

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        # self.running_metric = ConfuseMatrixMeter(n_class=args.n_class)
        self.running_metric = MetricScore(n_class=args.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_mf1 = 0
        self.best_val_mf1 = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        # define the loss functions
        if args.loss == 'ce':
            # self._pxl_loss = cross_entropy
            # self._pxl_loss = nn.CrossEntropyLoss()
            self._pxl_loss = nn.CrossEntropyLoss(ignore_index=0)
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        else:
            raise NotImplemented(args.loss)

        self.VAL_mF1 = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_mF1.npy')):
            self.VAL_mF1 = np.load(os.path.join(self.checkpoint_dir, 'val_mF1.npy'))
        self.TRAIN_mF1 = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_mf1.npy')):
            self.TRAIN_mF1 = np.load(os.path.join(self.checkpoint_dir, 'train_mf1.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_mf1 = checkpoint['best_val_mf1']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_mf1 = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_mf1, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _collect_running_batch_states(self, target, G_pred, G_loss):
        target = target.detach()
        G_pred = torch.argmax(G_pred.detach(), dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()

        if np.mod(self.batch_id, 100) == 1:
            if self.is_training is True:
                message = 'Is_training: [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, mf1: %.5f\n' % (self.epoch_id, self.max_num_epochs-1, self.batch_id, m, imps*self.batch_size, est, G_loss.item(), current_score)
            else:
                message = 'Is_validing: [%d,%d][%d,%d], imps: %.2f, est: %.2fh, mf1: %.5f\n' % (self.epoch_id, self.max_num_epochs-1, self.batch_id, m, imps*self.batch_size, est, current_score)
            self.logger.write(message)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_mf1 = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d:\nOA: %.5f, mF1: %.5f, mIoU: %.5f \n' % (self.is_training, self.epoch_id, self.max_num_epochs-1, scores['acc'], self.epoch_mf1, scores['miou']))
        message = 'Detial message: \n'
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_mf1': self.best_val_mf1,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_checkpoints(self):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_mf1=%.4f, Historical_best_mf1=%.4f (at epoch %d)\n' % (self.epoch_mf1, self.best_val_mf1, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_mf1 > self.best_val_mf1:
            self.best_val_mf1 = self.epoch_mf1
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_mF1 = np.append(self.TRAIN_mF1, [self.epoch_mf1])
        np.save(os.path.join(self.checkpoint_dir, 'train_mf1.npy'), self.TRAIN_mF1)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_mF1 = np.append(self.VAL_mF1, [self.epoch_mf1])
        np.save(os.path.join(self.checkpoint_dir, 'VAL_mF1.npy'), self.VAL_mF1)

    def train_models(self):
        self._load_checkpoint()
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            ################## train #################

            self.running_metric.reset()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                img = batch['image'].to(self.device)
                gt = batch['label'].to(self.device).long()
                if gt.dim() == 4:
                    gt = torch.squeeze(gt, dim=1)
                self.optimizer_G.zero_grad()
                output = self.net_G(img)
                if self.net_name == 'wetr' or self.net_name == 'segformer':
                    output = F.interpolate(output, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                G_pred = torch.argmax(output.detach(), dim=1)
                G_loss = self._pxl_loss(output, gt)
                G_loss.backward()
                self.optimizer_G.step()

                self.running_metric.update(gt.data.cpu().numpy(), G_pred.cpu().numpy())
                # self._collect_running_batch_states(gt, G_pred, G_loss)
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self.exp_lr_scheduler_G.step()

            ################## Eval ##################
            self.logger.write('Begin evaluation...\n')
            self.running_metric.reset()
            self.is_training = False
            self.net_G.eval()
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img = batch['image'].to(self.device)
                    gt = batch['label'].to(self.device).long()
                    if gt.dim() == 4:
                        gt = torch.squeeze(gt, dim=1)
                    # gt -= 1
                    output = self.net_G(img)
                    if self.net_name == 'wetr' or self.net_name == 'segformer':
                        output = F.interpolate(output, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                    G_pred = torch.argmax(output.detach(), dim=1)

                self.running_metric.update(gt.data.cpu().numpy(), G_pred.cpu().numpy())
                # self._collect_running_batch_states(gt, G_pred, 1)
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            self._update_val_acc_curve()
            self._update_checkpoints()

