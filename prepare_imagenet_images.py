# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import shutil
import argparse
import numpy as np

def parse_args():
    usage = 'Parser for generate biggan images script.'
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument(
        '--imagenet_dir', type=str, required=True, help='path to the imagenet folder')
    parser.add_argument(
        '--dataset_dir', type=str, default='./data/', help='path to the dataset dir')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    real_list_path = os.path.join(args.dataset_dir, 'annotations/real-random-list.txt')
    real_list = np.loadtxt(real_list_path, dtype=str)

    for file_list in real_list:
        class_n = file_list.split('/')[0]
        real_anno_n = file_list.split('/')[1]

        img_id = real_anno_n.split('_')[-1]
        imagenet_file_name = class_n + '_' + img_id + '.JPEG'
        # copy imagenet image to dataset
        imagenet_file_path = os.path.join(args.imagenet_dir, class_n, imagenet_file_name)
        real_image_dir = os.path.join(args.dataset_dir, 'images/real-random/', class_n)
        os.makedirs(real_image_dir, exist_ok=True)
        save_path = os.path.join(real_image_dir, real_anno_n + '.jpg')
        print('Copy image from {0} to {1}'.format(imagenet_file_path, save_path))
        shutil.copy(imagenet_file_path, save_path)
  
