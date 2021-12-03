import os
import os.path as osp
from PIL import Image
import numpy as np
import pdb
from torch.utils.data import Dataset
from datasets.data_utils import DataAugmentation

class SegDataset(Dataset):

    def __init__(self, root_dir, img_size, data_str, split='train', is_train=True, transform=None, to_tensor=True):
        super(SegDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.data_str = data_str
        self.split = split
        self.to_tensor = to_tensor
        self.transform = transform
        self.img_list = self.load_img_list(osp.join(self.root_dir, 'list', self.split + '.txt'))
        self.num_img = len(self.img_list)  # get the size of dataset A
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size)

    def __getitem__(self, index):
        name = self.img_list[index]
        img_path = osp.join(self.root_dir, self.data_str[1], self.split, self.img_list[index % self.num_img] + ".png")
        # img = np.asarray(Image.open(img_path).convert('RGB'))
        img = Image.open(img_path).convert('RGB')

        # dsm_path = osp.join(self.root_dir, self.data_str[0], self.split, self.img_list[index % self.num_img].replace('label', 'dsm') + ".tif")
        # dsm = np.asarray(Image.open(dsm_path))

        Label_path = osp.join(self.root_dir, self.data_str[2], self.split, self.img_list[index % self.num_img] + ".png")
        # label = np.array(Image.open(Label_path), dtype=np.uint8)
        label = Image.open(Label_path)
        # label += 1      # 切片的切片标签为0
        sample = {'image': img, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample = self.img2np(sample, name)
        # img, label = self.augm.transform(img, label, to_tensor=self.to_tensor)
        return sample
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.num_img
    
    def load_img_list(self, dataset_path):
        img_list = np.loadtxt(dataset_path, dtype=np.str)
        if img_list.ndim == 2:
            return img_list[:, 0]
        return img_list
    
    def img2np(self, sample, name):
        sample['image'] = np.asarray(sample['image'])
        sample['label'] = np.asarray(sample['label'], dtype=np.uint8)
        sample['name'] = name
        return sample


