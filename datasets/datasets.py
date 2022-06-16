# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImagenetDataset(Dataset):
    # dataset for training bigdataset gan
    def __init__(self, data_root):
        self.label_dir = os.path.join(data_root, 'annotations/biggan512/')
        self.latent_dir = os.path.join(data_root, 'latents/biggan512/')

        self._prepare_data_list()

    def _prepare_data_list(self):
        class_list = sorted(os.listdir(self.label_dir))
        label_list = []
        latent_list = []
        for class_n in class_list:
            label_file_list = sorted(os.listdir(os.path.join(self.label_dir, class_n)))
            latent_file_list = sorted(os.listdir(os.path.join(self.latent_dir, class_n)))

            for label_file_n, latent_file_n in zip(label_file_list, latent_file_list):
                label_list.append(os.path.join(class_n, label_file_n))
                latent_list.append(os.path.join(class_n, latent_file_n))
        
        self.label_list = label_list
        self.latent_list = latent_list
        self.data_size = len(self.label_list)
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        latent_z = np.load(os.path.join(self.latent_dir, self.latent_list[idx]))[0]
        label_pil = Image.open(os.path.join(self.label_dir, self.label_list[idx])).convert('L')
        label_np = np.array(label_pil)
        # make label to 1
        label_np[label_np != 0] = 1
        class_y = int(self.label_list[idx].split('.')[0].split('_')[-2])

        latent_z = torch.tensor(latent_z, dtype=torch.float)
        label_tensor = torch.tensor(label_np, dtype=torch.long)
        class_y_tensor = torch.tensor(class_y, dtype=torch.long)

        return {
            'latent': latent_z,
            'label': label_tensor,
            'y': class_y_tensor,
        }

