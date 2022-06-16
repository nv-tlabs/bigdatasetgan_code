# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from models import BigdatasetGANModel
import numpy as np
import argparse
import torch
import torchvision

def parse_args():
    usage = 'Parser for generate biggan images script.'
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument(
        '--biggan_ckpt', type=str, default='./pretrain/biggan-512.pth', help='path to the pretrained biggan ckpt')
    parser.add_argument(
        '--dataset_dir', type=str, default='./data/', help='path to the dataset dir')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda'

    latents_dir = os.path.join(args.dataset_dir, 'latents/biggan512/')
    images_dir = os.path.join(args.dataset_dir, 'images/biggan512/')

    # loading model
    generator = BigdatasetGANModel(resolution=512, out_dim=1, biggan_ckpt=args.biggan_ckpt).to(device)

    generator.eval()

    class_list = os.listdir(latents_dir)

    for class_n in class_list:
        latent_class_dir = os.path.join(latents_dir, class_n)
        image_class_dir = os.path.join(images_dir, class_n)

        os.makedirs(image_class_dir, exist_ok=True)

        latent_list = os.listdir(latent_class_dir)

        for latent_n in latent_list:
            image_name = latent_n.split('.')[0]
            latent_np = np.load(os.path.join(latent_class_dir, latent_n))[0]
            class_y = int(image_name.split('_')[-2])

            latent_tensor = torch.tensor(latent_np, dtype=torch.float).unsqueeze(0).to(device)
            class_y_tensor = torch.tensor([class_y], dtype=torch.long).to(device)
            
            image_tensor, _ = generator.sample(latent_tensor, class_y_tensor)
            
            print("Saving biggan images from the latent to: ", os.path.join(image_class_dir, image_name+'.png'))
            # save image
            torchvision.utils.save_image(image_tensor, os.path.join(image_class_dir, image_name+'.png'), normalize=True)
        