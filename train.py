# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch
import torch.nn as nn
import argparse
import torchvision

import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import BigdatasetGANModel
from datasets.datasets import ImagenetDataset
from utils import VOCColorize

def parse_args():
    usage = 'Parser for training bigdataset script.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        '--resolution', '-r', type=int, default=512,
        help='Resolution of the generated images, we use biggan-512 by default')

    parser.add_argument(
        '--gan_ckpt', type=str, default='./pretrain/biggan-512.pth', 
        help='Path to the pretrained gan ckpt')
    parser.add_argument(
        '--dataset_dir', type=str, default='./data/',
        help='Path to the dataset folder')
    
    parser.add_argument(
        '--save_dir', type=str, default='./logs/checkpoint_biggan512_label_conv/',
        help='Path to save logs')
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='training batch size')
    parser.add_argument(
        '--max_iter', type=int, default=5000,
        help='maximum iteration of training')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='learning rate')

    args = parser.parse_args()

    return args

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def main(args):
    device = 'cuda'
    # build checkpoint dir
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_dir = os.path.join(args.save_dir, 'run-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    # os.makedirs(os.path.join(ckpt_dir, 'samples'), exist_ok=True)

    # build dataset
    dataset = ImagenetDataset(args.dataset_dir)
    #dataset = BigganDataset(args.dataset_dir, single_class=args.single_class)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("loading dataset size: ", len(dataset))
    
    # build seg model
    g_seg = BigdatasetGANModel(resolution=args.resolution, out_dim=1, biggan_ckpt=args.gan_ckpt)

    g_seg = g_seg.to(device)

    g_optim = optim.Adam(
                g_seg.parameters(),
                lr=args.lr
              )

    loss_fn = nn.BCEWithLogitsLoss()

    dataloader = sample_data(dataloader)

    voc_col = VOCColorize(n=1000)

    print("Start training with maximum {0} iterations.".format(args.max_iter))

    for i, batch_data in enumerate(dataloader):

        if i > args.max_iter:
            break

        z = batch_data['latent'].to(device)
        label_gt = batch_data['label'].to(device)
        y = batch_data['y'].to(device)
    
        # set g_Seg in train mode
        g_seg.train()
        g_seg.biggan_model.eval()

        label_pred = g_seg(z, y)

        loss = loss_fn(label_pred, label_gt.float().unsqueeze(1))
     
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        
        writer.add_scalar('train/loss', loss.item(), global_step=i)

        if i % 10 == 0:
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}".format(i, args.max_iter, loss))

        if i % 100 == 0:
            # save train pred
            g_seg.eval()
            sample_imgs, sample_pred = g_seg.sample(z, y)
            sample_imgs, sample_pred = sample_imgs.cpu(), sample_pred.cpu()

            label_pred_prob = torch.sigmoid(label_pred)
            label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.long)
            label_pred_mask[label_pred_prob>0.5] = 1

            label_pred_rgb = voc_col(label_pred_mask[0][0].cpu().numpy()*y[0].cpu().numpy())
            label_pred_rgb = torch.from_numpy(label_pred_rgb).float()

            label_gt_rgb = voc_col(label_gt[0].cpu().numpy()*y[0].cpu().numpy())
            label_gt_rgb = torch.from_numpy(label_gt_rgb).float()

            viz_tensor = torch.stack([sample_imgs[0], label_gt_rgb, label_pred_rgb], dim=0)

            torchvision.utils.save_image(viz_tensor, os.path.join(ckpt_dir, 
                                                'training/viz_sample_{0:05d}.jpg'.format(i)), normalize=True, scale_each=True)

        if i % 1000 == 0:
            # save checkpoint
            print("Saving latest checkpoint.")
            torch.save(g_seg.state_dict(), os.path.join(ckpt_dir, 'checkpoint_latest.pth'))

if __name__ == '__main__':

    args = parse_args()

    main(args)
    