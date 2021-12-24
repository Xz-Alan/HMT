from argparse import ArgumentParser
from urllib import parse
import torch
from models.trainer import *
from data_config import DataConfig
# warnings.filterwarnings("ignore")

print(torch.cuda.is_available())

def train(args):
    dataloaders = utils.get_loaders(args)
    model = SegTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    from models.evaluator import SegEvaluator
    dataloader = utils.get_loader(args.root_dir, args.data_str, img_size=args.img_size, 
                                batch_size=args.batch_size, is_train=False, split='test')
    model = SegEvaluator(args=args, dataloader=dataloader)
    model.eval_models()

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='5,6', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='MFT', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--root_dir', default='', type=str)
    parser.add_argument('--data_str', default=[], type=list)
    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='SegDataset', type=str)
    parser.add_argument('--data_name', default='Vaihingen', type=str, help='Potsdam | Vaihingen | GID | quick_start')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="valid", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|pspnet')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd | adam')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--lr_policy', default='step', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--palette_path', default='../data/palette.json', type=str)

    args = parser.parse_args()
    utils.get_device(args)

    dataConfig = DataConfig().get_data_config(args.data_name)
    args.root_dir = dataConfig.root_dir
    args.data_str = dataConfig.data_str
    args.n_class = dataConfig.num_class
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("start training")
    train(args)
    print("finish training")
    print("-----------------------------------------------------------------------")
    print("start testing")
    args.split = 'test'
    test(args)
    print("finish testing")