# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch
import argparse
import torchvision

from models import BigdatasetGANModel
from utils import VOCColorize

def parse_args():
    usage = 'Parser for training bigdataset script.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        '--resolution', '-r', type=int, default=512,
        help='Resolution of the generated images, we use biggan-512 by default')
    parser.add_argument(
        '--ckpt', type=str, required=True, 
        help='Path to the pretrained BigDatasetGAN weights')
    parser.add_argument(
        '--save_dir', type=str, default='./generated_datasets/',
        help='Path to save dataset')
    parser.add_argument(
        '--z_var', type=float, default=0.9,
        help='Truancation value of z')
    parser.add_argument(
        '--class_idx', type=int, default=[225, 200], nargs='+',
        help='Imagenet class index')
    parser.add_argument(
        '--samples_per_class', type=int, default=10,
        help='data samples per class')

    args = parser.parse_args()

    return args

def main(args):
    device = 'cuda'
    
    # build seg model
    model = BigdatasetGANModel(resolution=args.resolution, out_dim=1, biggan_ckpt=None)

    # load pretrain model
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict, strict=False) # Ignore missing sv0 entries

    model = model.to(device)
    model = model.eval()

    voc_col = VOCColorize(n=1000)

    overall_viz = []

    os.makedirs(args.save_dir, exist_ok=True)

    for class_y in args.class_idx:
        print("Start sampling dataset with class idx: {0}, total samples: {1}".format(class_y, args.samples_per_class))

        class_y_tensor = torch.tensor([class_y], dtype=torch.long).to(device)

        sample_imgs, sample_labels = [], []
        for i in range(args.samples_per_class):
            z = torch.empty(1, model.biggan_model.dim_z).normal_(mean=0, std=args.z_var).to(device)
            sample_img, sample_pred = model.sample(z, class_y_tensor)
            sample_img, sample_pred = sample_img.cpu(), sample_pred.cpu()

            label_pred_prob = torch.sigmoid(sample_pred)
            label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.long)
            label_pred_mask[label_pred_prob>0.5] = 1

            label_pred_rgb = voc_col(label_pred_mask[0][0].cpu().numpy()*class_y)
            label_pred_rgb = torch.from_numpy(label_pred_rgb).float()

            sample_imgs.append(sample_img)
            sample_labels.append(label_pred_rgb)
        
        sample_imgs = torch.cat(sample_imgs, dim=0)
        sample_labels = torch.stack(sample_labels, dim=0)

        sample_imgs_grid = torchvision.utils.make_grid(sample_imgs, nrow=args.samples_per_class, normalize=True, scale_each=True)
        sample_labels_grid = torchvision.utils.make_grid(sample_labels, nrow=args.samples_per_class, normalize=True, scale_each=True)
        class_viz_tensor = torchvision.utils.make_grid(torch.stack([sample_imgs_grid, sample_labels_grid]), dim=0, nrow=1)
        overall_viz.append(class_viz_tensor)
    
    overall_viz = torch.stack(overall_viz, dim=0)
    torchvision.utils.save_image(overall_viz, os.path.join(args.save_dir, 'sample_overall.jpg'), nrow=1)

if __name__ == '__main__':

    args = parse_args()

    main(args)
    