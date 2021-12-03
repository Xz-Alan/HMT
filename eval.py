from argparse import ArgumentParser
import torch
from models.evaluator import *
from data_config import DataConfig
print(torch.cuda.is_available())

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='SegDataset', type=str)
    parser.add_argument('--data_name', default='Vaihingen', type=str, help='Potsdam | Vaihingen | GID | quick_start')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    parser.add_argument('--palette_path', default='../data/palette.json', type=str)

    args = parser.parse_args()
    utils.get_device(args)

    dataConfig = DataConfig().get_data_config(args.data_name)
    args.root_dir = dataConfig.root_dir
    args.data_str = dataConfig.data_str
    args.n_class = dataConfig.num_class

    #  checkpoints dir
    args.checkpoint_dir = os.path.join('checkpoints', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.root_dir, args.data_str, img_size=args.img_size, 
                            batch_size=args.batch_size, is_train=False, split=args.split)
    print("start evaluation")
    model = SegEvaluator(args=args, dataloader=dataloader)
    model.eval_models(checkpoint_name=args.checkpoint_name)
    print("finish evaluation")

if __name__ == '__main__':
    main()

