# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./biggan_pytorch')
from biggan_pytorch import BigGAN

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                which_conv=nn.Conv2d, which_linear=None, activation=None, 
                upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = BigGAN.layers.ccbn(in_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        self.bn2 = BigGAN.layers.ccbn(out_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x

def get_config(resolution):
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}
    config = {'G_param': 'SN', 'D_param': 'SN', 
            'G_ch': 96, 'D_ch': 96, 
            'D_wide': True, 'G_shared': True, 
            'shared_dim': 128, 'dim_z': dim_z_dict[resolution], 
            'hier': True, 'cross_replica': False, 
            'mybn': False, 'G_activation': nn.ReLU(inplace=True),
            'G_attn': attn_dict[resolution],
            'norm_style': 'bn',
            'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
            'G_fp16': False, 'G_mixed_precision': False,
            'accumulate_stats': False, 'num_standing_accumulations': 16, 
            'G_eval_mode': True,
            'BN_eps': 1e-04, 'SN_eps': 1e-04, 
            'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution, 
            'n_classes': 1000}
    return config

class BigdatasetGANModel(nn.Module):
    def __init__(self, resolution, out_dim, biggan_ckpt=None):
        super(BigdatasetGANModel, self).__init__()
        self.biggan_ckpt = biggan_ckpt
        self.resolution = resolution
        # load biggan model
        self._prepare_biggan_model()

        self.low_feature_size = 32
        self.mid_feature_size = 128
        self.high_feature_size = 512
        
        low_feature_channel = 128
        mid_feature_channel = 64
        high_feature_channel = 32

        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(3072, low_feature_channel, kernel_size=1, bias=False),
            #nn.ReLU(),
        )
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d(960, mid_feature_channel, kernel_size=1, bias=False),
            #nn.ReLU(),
        )
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel,
                                out_channels=low_feature_channel+mid_feature_channel,
                                con_channels=self.biggan_model.shared_dim,
                                which_conv=self.biggan_model.which_conv,
                                which_linear=self.biggan_model.which_linear,
                                activation=self.biggan_model.activation,
                                upsample=False,
                            )

        self.high_feature_conv = nn.Sequential(
            nn.Conv2d(192, high_feature_channel, kernel_size=1, bias=False),
            #nn.ReLU(),
        )

        self.high_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel+high_feature_channel,
                                out_channels=low_feature_channel+mid_feature_channel+high_feature_channel,
                                con_channels=self.biggan_model.shared_dim,
                                which_conv=self.biggan_model.which_conv,
                                which_linear=self.biggan_model.which_linear,
                                activation=self.biggan_model.activation,
                                upsample=False,
                            )

        self.out_layer = nn.Conv2d(low_feature_channel+mid_feature_channel+high_feature_channel, 
                                    out_dim, kernel_size=3, padding=1)
        self.out_layer = nn.Sequential(
                                BigGAN.layers.bn(low_feature_channel+mid_feature_channel+high_feature_channel),
                                self.biggan_model.activation,
                                self.biggan_model.which_conv(low_feature_channel+mid_feature_channel+high_feature_channel, out_dim)
                            )

    def _prepare_biggan_model(self):
        biggan_config = get_config(self.resolution)
        self.biggan_model = BigGAN.Generator(**biggan_config)
        if self.biggan_ckpt != None:
            state_dict = torch.load(self.biggan_ckpt)
            self.biggan_model.load_state_dict(state_dict, strict=False) # Ignore missing sv0 entries
        self.biggan_model.eval()

    def _prepare_features(self, features, upsample='bilinear'):
        # for low feature
        low_features = [
            F.interpolate(features[0], size=self.low_feature_size, mode=upsample, align_corners=False),
            F.interpolate(features[1], size=self.low_feature_size, mode=upsample, align_corners=False),
            features[2],
        ]
        low_features = torch.cat(low_features, dim=1)
        # for mid feature
        mid_features = [
            F.interpolate(features[3], size=self.mid_feature_size, mode=upsample, align_corners=False),
            F.interpolate(features[4], size=self.mid_feature_size, mode=upsample, align_corners=False),
            features[5],
        ]
        mid_features = torch.cat(mid_features, dim=1)
        # for high feature
        high_features = [
            F.interpolate(features[6], size=self.high_feature_size, mode=upsample, align_corners=False),
            #F.interpolate(features[7], size=self.high_feature_size, mode=upsample, align_corners=False),
            features[7],
        ]
        high_features = torch.cat(high_features, dim=1)
        
        features_dict = {
            'low': low_features,
            'mid': mid_features,
            'high': high_features,
        }

        return features_dict

    @torch.no_grad()
    def _get_biggan_features(self, z, y):
        features = []
        y = self.biggan_model.shared(y)
        # forward thru biggan
        # If hierarchical, concatenate zs and ys
        if self.biggan_model.hier:
            zs = torch.split(z, self.biggan_model.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.biggan_model.blocks)
        
        # First linear layer
        h = self.biggan_model.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.biggan_model.bottom_width, self.biggan_model.bottom_width)
        
        # Loop over blocks
        for index, blocklist in enumerate(self.biggan_model.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])
                # save feature
                features.append(h)
                #print(index, h.shape)

        features_dict = self._prepare_features(features)

        return features_dict, y, h

    def forward(self, z, y):
        features_dict, y, _ = self._get_biggan_features(z, y)

        # for low features
        low_feat = self.low_feature_conv(features_dict['low'])
        low_feat = F.interpolate(low_feat, size=self.mid_feature_size, mode='bilinear', align_corners=False)
        # for mid features
        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat, y)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)
        # for high features
        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y)
        out = self.out_layer(high_feat)

        return out

    @torch.no_grad()
    def sample(self, z, y):
        features_dict, y, h = self._get_biggan_features(z,y)

        image = torch.tanh(self.biggan_model.output_layer(h.detach()))
        
        # for low features
        low_feat = self.low_feature_conv(features_dict['low'])
        low_feat = F.interpolate(low_feat, size=self.mid_feature_size, mode='bilinear', align_corners=False)
        # for mid features
        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat, y)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)
        # for high features
        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y)
        out = self.out_layer(high_feat)
        
        return image, out


if __name__ == '__main__':

    biggan_ckpt = './pretrain/biggan-512.pth'
    model = BigdatasetGANModel(512, 1, biggan_ckpt).cuda()