'''
saliency data loader

@author: Hamed R. Tavakoli
'''

import os

import random
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from torch.utils.data import Dataset
from utils import padded_resize, resize_padded_fixation


class SalDB(Dataset):
    '''
        class to load the images
    '''
    def __init__(self, root_folder, fold, input_size, output_size, input_transform=None, target_transform=None, blur_sigma=None):
        super(SalDB, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.target_transform = target_transform
        self.input_transform = input_transform
        self.img_folder = os.path.join(root_folder, 'images', fold)
        self.map_folder = os.path.join(root_folder, 'maps', fold)
        self.fix_folder = os.path.join(root_folder, 'fixmaps', fold)
        self.blur_sigma = blur_sigma
        self.item = []

        for file in os.listdir(self.img_folder):
            if file.endswith('png'):
                img_id = file[:-4]
                self.item.append(img_id)

    def __len__(self):
        return len(self.item)

    def __getitem__(self, index):
        img_id = self.item[index]
        img = Image.open(os.path.join(self.img_folder, '{}.png'.format(img_id))).convert('RGB')
        map = Image.open(os.path.join(self.map_folder, '{}.png'.format(img_id))).convert('L')
        fix = Image.open(os.path.join(self.fix_folder, '{}.png'.format(img_id))).convert('L')

        if self.blur_sigma is not None:
            img = img.filter(ImageFilter.GaussianBlur(self.blur_sigma))

        img = padded_resize(img, self.input_size[0], self.input_size[1])
        map = padded_resize(map, self.output_size[0], self.output_size[1])
        fix = resize_padded_fixation(fix, self.output_size[0], self.output_size[1])

        c = random.random()
        if c > 0.5:
            img = ImageOps.flip(img)
            map = ImageOps.flip(map)
            fix = ImageOps.flip(fix)

        h = random.random()
        if h > 0.5:
            img = ImageOps.mirror(img)
            map = ImageOps.mirror(map)
            fix = ImageOps.mirror(fix)

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.target_transform is not None:
            map = self.target_transform(map)
            fix = self.target_transform(fix)

        return img_id, img, map, fix

