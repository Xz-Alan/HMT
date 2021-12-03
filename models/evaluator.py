import json
import os
import os.path as osp
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import utils
from misc.logger_tool import Logger
from misc.metric_tool import ConfuseMatrixMeter, MetricScore
from utils import de_norm

from models.init_networks import *

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SegEvaluator():

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        self.split = args.split
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_name = args.net_G
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0 else "cpu")
        print(self.device)

        # define some other vars to record the training states
        # self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        self.running_metric = MetricScore(n_class=args.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_mf1 = 0
        self.best_val_mf1 = 0.0
        self.best_epoch_id = 0
        self.is_training = False
        self.batch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        # check and create model dir
        self.vis_dir = args.vis_dir
        self.palette_path = args.palette_path
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_mf1 = checkpoint['best_val_mf1']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_mf1 = %.4f (at epoch %d)\n' %
                  (self.best_val_mf1, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _collect_running_batch_states(self, target, G_pred):
        target = target.detach()
        G_pred = torch.argmax(G_pred.detach(), dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        m = len(self.dataloader)
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d], mf1: %.5f\n' % (self.is_training, self.batch_id, m, current_score)
            self.logger.write(message)

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_mf1 = scores_dict['mf1']
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_mf1)), mode='a') as file:
            pass
        message = 'Final message: \n'
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')  # save the message
        self.logger.write('\n')

    def pred2vis(self, pred):
        with open(self.palette_path, 'r') as fp:
            text = json.load(fp)
        list_value = np.asarray(list(text.values()), dtype=np.uint8)
        vis_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                vis_pred[i,j,:] = list_value[pred[i,j]]
        return vis_pred

    def eval_models(self,checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        ################## Eval ##################
        self.logger.write('Begin evaluation...\n')
        self.running_metric.reset()
        self.is_training = False
        self.net_G.eval()
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                img = batch['image'].to(self.device)
                gt = batch['label'].to(self.device).long()
                if gt.dim() == 4:
                    gt = torch.squeeze(gt, dim=1)
                output = self.net_G(img)
                if self.net_name == 'wetr' or self.net_name == 'segformer':
                    output = F.interpolate(output, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                G_pred = torch.argmax(output.detach(), dim=1)
                if self.split == 'test':
                    for i in range(len(batch['name'])):
                        temp_path = osp.join(self.vis_dir, batch['name'][i] + '.png')
                        temp_img = self.pred2vis(G_pred[i].cpu().numpy().astype(np.uint8))
                        cv2.imwrite(temp_path, temp_img)
                    # pdb.set_trace()
            # self._collect_running_batch_states(gt, G_pred)
            self.running_metric.update(gt.data.cpu().numpy(), G_pred.cpu().numpy())
        self._collect_epoch_states()
