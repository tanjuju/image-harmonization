import os.path
import torch
import random
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class Iharmony4Dataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt, is_for_train):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.isTrain = is_for_train
        self._load_images_paths()
        self.transform = get_transform(opt)

    def _load_images_paths(self,):
        if self.isTrain == True:
            print('loading training file...')
            self.trainfile = os.path.join(self.opt.dataset_root, 'IHD_train.txt')######################原来的
            # self.trainfile = os.path.join(self.opt.dataset_root, 'HAdobe5k_train.txt')
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    #注释 高分辨率
                    data_parts = line.split('/')
                    if data_parts[0] != 'HAdobe5k':
                       continue

                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    '''
                    line='composite_images/'+line
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_' + name_parts[-2] + '_' + name_parts[-1], '.jpg')
                    '''


                    
                    self.image_paths.append(os.path.join(self.opt.dataset_root, line))
                    self.mask_paths.append(os.path.join(self.opt.dataset_root, mask_path))
                    self.gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))

        elif self.isTrain == False:
            print('loading test file...')
            # self.trainfile = os.path.join(self.opt.dataset_root, 'test.txt')
            self.trainfile = os.path.join(self.opt.dataset_root, 'IHD_test.txt')########原来的
            # self.trainfile = os.path.join(self.opt.dataset_root, 'HAdobe5k_test.txt')#############用real99数据集的
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    data_parts = line.split('/')

                    ##高分辨率
                    if data_parts[0] != 'Hday2night':
                       continue

                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    '''
                    line = 'composite_images/' + line
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_' + name_parts[-2] + '_' + name_parts[-1], '.jpg')
                    '''

                    
                    self.image_paths.append(os.path.join(self.opt.dataset_root, line))
                    self.mask_paths.append(os.path.join(self.opt.dataset_root, mask_path))
                    self.gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert('RGB')#0-255
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('1')#0或255

        comp = tf.resize(comp, [256, 256])
        mask = tf.resize(mask, [256, 256])
        real = tf.resize(real, [256, 256])
        '''
        comp = tf.resize(comp, [1024, 1024])
        mask = tf.resize(mask, [1024, 1024])
        real = tf.resize(real, [1024, 1024])
        '''

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        mask = tf.to_tensor(mask)#除以255 转换0-1之间
        real = self.transform(real)#-1  1
        comp = self._compose(comp, mask, real)

        return {'comp': comp, 'mask': mask, 'real': real, 'img_path':self.image_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def _compose(self, foreground_img, foreground_mask, background_img):
        return foreground_img * foreground_mask + background_img * (1 - foreground_mask)
