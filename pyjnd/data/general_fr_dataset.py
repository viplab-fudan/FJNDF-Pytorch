from PIL import Image
from os import path as osp

import torch
from torch.utils import data as data
import torchvision.transforms as tf
from torchvision.transforms.functional import to_pil_image

import numpy as np
import cv2
from pyjnd.data.transforms import transform_mapping, PairedToTensor
from pyjnd.utils.registry import DATASET_REGISTRY
from pyjnd.utils import yuvread2tensor

from .base_jnd_dataset import BaseJNDDataset

@DATASET_REGISTRY.register()
class GeneralFRDataset(BaseJNDDataset):
    """General Full Reference dataset with meta info file.
    """
    
    def init_path_jnd(self, opt):
        super().init_path_jnd(opt)

        input_img_folder = opt['dataroot_in']
        ref_img_folder = opt.get('dataroot_ref', None)
        if ref_img_folder is None:
            ref_img_folder = input_img_folder

        cols = list(self.meta_info.columns)
        idx_input  = cols.index('input_image')
        idx_ref    = cols.index('reference_image')
        idx_jnd    = cols.index('jnd_mean')
        idx_width  = cols.index('yuv_width')
        idx_height = cols.index('yuv_height')
        idx_format = cols.index('yuv_format')
        idx_bitdp  = cols.index('yuv_bitdepth')

        self.paths_jnd = []
        for row in self.meta_info.values:
            input_path = osp.join(input_img_folder, row[idx_input])
            ref_path = osp.join(ref_img_folder, row[idx_ref])
            jnd_label = float(row[idx_jnd])
            yuv_w = int(row[idx_width ])
            yuv_h = int(row[idx_height])
            yuv_fmt = row[idx_format]
            yuv_bd = int(row[idx_bitdp ])
            self.paths_jnd.append([input_path, ref_path, jnd_label, yuv_w, yuv_h, yuv_fmt, yuv_bd])

    def get_transforms(self, opt):
        # do paired transform first and then do common transform
        paired_transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                paired_transform_list += transform_mapping(k, v)
        self.paired_trans = tf.Compose(paired_transform_list)

        common_transform_list = []
        self.img_range = opt.get('img_range', 1.0)
        common_transform_list += [
                PairedToTensor(),
                ]
        self.common_trans = tf.Compose(common_transform_list)
    
    def jnd_normalize(self, opt):
        jnd_range = opt.get('jnd_range', None)
        jnd_lower_better = opt.get('lower_better', None)
        jnd_normalize = opt.get('jnd_normalize', False)

        if jnd_normalize:
            assert jnd_range is not None and jnd_lower_better is not None, 'jnd_range and jnd_lower_better should be provided when jnd_normalize is True'

            def normalize(jnd_label):
                jnd_label = (jnd_label - jnd_range[0]) / (jnd_range[1] - jnd_range[0])
                if jnd_lower_better:
                    jnd_label = 1 - jnd_label
                return jnd_label

            self.paths_jnd = [item[:2] + [normalize(item[2])] for item in self.paths_jnd]
            self.logger.info(f'jnd_label is normalized from {jnd_range}, lower_better[{jnd_lower_better}] to [0, 1], higher better.')

    def __getitem__(self, index):
        input_path = self.paths_jnd[index][0]
        ref_path = self.paths_jnd[index][1]
        jnd_label = self.paths_jnd[index][2]

        ext = osp.splitext(ref_path)[1].lower()
        if ext == '.yuv':
            w       = self.paths_jnd[index][3]
            h       = self.paths_jnd[index][4]
            fmt     = self.paths_jnd[index][5]
            bd      = self.paths_jnd[index][6]
            img_tensor = yuvread2tensor(input_path, w, h, fmt=fmt, bitdepth=bd, normalize=True)
            ref_tensor = yuvread2tensor(ref_path, w, h, fmt=fmt, bitdepth=bd, normalize=True)
            img_pil = to_pil_image(img_tensor)
            ref_pil = to_pil_image(ref_tensor)
        else:
            img_pil = Image.open(input_path).convert('RGB')
            ref_pil = Image.open(ref_path).convert('RGB')
        
        # 3. paired/random transform (assume self.paired_trans supports Tensor)
        img_pil, ref_pil = self.paired_trans([img_pil, ref_pil])
        
        # 4. common transform (such as ToTensor/range scaling)
        img_tensor = self.common_trans(img_pil) * self.img_range
        ref_tensor = self.common_trans(ref_pil) * self.img_range

        jnd_label_tensor = torch.Tensor([jnd_label])
        return {'img': img_tensor, 'ref_img': ref_tensor, 'jnd_label': jnd_label_tensor, 'img_path': input_path, 'ref_path': ref_path}