# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .imagenet_utils import pascal_to_synset, pascal_to_id, imagenet_to_synset, random_synset_100
import json
import itertools
from torchvision import datasets, transforms

class ImagenetDataset(Dataset):
    def __init__(self, data_root):
        #self.image_dir = os.path.join(data_root, 'image')
        self.label_dir = os.path.join(data_root, 'label')
        self.latent_dir = os.path.join(data_root, 'latent')

        #self.image_list = sorted(os.listdir(self.image_dir))
        self.latent_list = sorted(os.listdir(self.latent_dir))
        self.label_list = sorted(os.listdir(self.label_dir))

        with open(os.path.join(data_root, 'classes.json'), 'r') as f:
            self.class_map = json.load(f)
     
        self.data_size = len(self.latent_list)

        self.transforms = transforms

    def _find_synset(self, y):
        synset_id = imagenet_to_synset[str(y)]
        return synset_id
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        #img_pil = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert('RGB')
        latent_z = np.load(os.path.join(self.latent_dir, self.latent_list[idx]))[0]
        label_np = np.load(os.path.join(self.label_dir, self.label_list[idx]))
        class_y = int(self.label_list[idx].split('.')[0].split('_')[-2])

        latent_z = torch.tensor(latent_z, dtype=torch.float)
        label_tensor = torch.tensor(label_np, dtype=torch.long)
        class_y_tensor = torch.tensor(class_y, dtype=torch.long)

        return {
            'latent': latent_z,
            'label': label_tensor,
            'y': class_y_tensor,
        }

class ImagenetDogDataset(ImagenetDataset):
    def __init__(self, data_root):
        super().__init__(data_root)
        self._filter_index()
        self.data_size = len(self.label_list)

    def _filter_index(self):
        ids = []
        filter_list = pascal_to_synset['dog']
        for i, img_n in enumerate(self.label_list):
            synset_id = img_n.split('.')[0].split('_')[-3]
            for synset_filter in filter_list:
                if synset_id in synset_filter:
                    ids.append(i)
                    break
        
        #self.image_list = list(np.array(self.image_list)[ids])
        self.latent_list = list(np.array(self.latent_list)[ids])
        self.label_list = list(np.array(self.label_list)[ids])

class ImagenetBirdDataset(ImagenetDataset):
    def __init__(self, data_root):
        super().__init__(data_root)
        self._filter_index()
        self.data_size = len(self.label_list)

    def _filter_index(self):
        ids = []
        filter_list = pascal_to_synset['bird']
        for i, img_n in enumerate(self.label_list):
            synset_id = img_n.split('.')[0].split('_')[-3]
            for synset_filter in filter_list:
                if synset_id in synset_filter:
                    ids.append(i)
                    break
        
        self.latent_list = list(np.array(self.latent_list)[ids])
        self.label_list = list(np.array(self.label_list)[ids])

class ImagenetPascalDataset(ImagenetDataset):
    def __init__(self, data_root):
        super().__init__(data_root)
        self._filter_index()
        self.data_size = len(self.label_list)

    def _filter_index(self):
        ids = []
        filter_list = [pascal_to_synset[key] for key in pascal_to_synset.keys()]
        filter_list = list(itertools.chain(*filter_list))
        
        for i, img_n in enumerate(self.label_list):
            synset_id = img_n.split('.')[0].split('_')[-3]
            for synset_filter in filter_list:
                if synset_id in synset_filter:
                    ids.append(i)
                    break
        
        self.latent_list = list(np.array(self.latent_list)[ids])
        self.label_list = list(np.array(self.label_list)[ids])

    def _find_pascal(self, synset_id):
        for key in pascal_to_synset.keys():
            for synset_filter in pascal_to_synset[key]:
                if synset_id in synset_filter:
                    return key
        
        return None

    def _remap_class(self, label_np, synset_id):
        pascal_id = self._find_pascal(synset_id)
        assert pascal_id != None

        label_id = pascal_to_id[pascal_id]
        label_np[label_np != 0] = label_id
        return label_np

class ImagenetRandom100Dataset(ImagenetDataset):
    def __init__(self, data_root):
        super().__init__(data_root)
        self._filter_index()
        self.data_size = len(self.label_list)

    def _filter_index(self):
        ids = []
        synset_filter = random_synset_100
        for i, img_n in enumerate(self.label_list):
            synset_id = img_n.split('.')[0].split('_')[-3]
            if synset_id in synset_filter:
                ids.append(i)
        
        self.latent_list = list(np.array(self.latent_list)[ids])
        self.label_list = list(np.array(self.label_list)[ids])

    def _find_class_id(self, synset_id):
        for i, id in enumerate(random_synset_100):
            if synset_id == id:
                return i
        
        return -1

    def _remap_class(self, label_np, synset_id):
        class_id = self._find_class_id(synset_id)
        assert class_id != None

        label_np[label_np != 0] = class_id
        return label_np


class ImagenetClass992Dataset(ImagenetDataset):
    def __init__(self, data_root):
        super().__init__(data_root)
        with open('./imagenetclass992.json', 'r') as f:
            self.class_dict = json.load(f)
        
        self._filter_index()

        self.data_size = len(self.label_list)

    def _filter_index(self):
        ids = []
        synset_filter = self.class_dict.keys()

        for i, img_n in enumerate(self.label_list):
            synset_id = img_n.split('.')[0].split('_')[-3]
            if synset_id in synset_filter:
                ids.append(i)
        
        self.latent_list = list(np.array(self.latent_list)[ids])
        self.label_list = list(np.array(self.label_list)[ids])

    def _find_class_id(self, synset_id):
        return self.class_dict[synset_id]['seg_id']

    def _remap_class(self, label_np, synset_id):
        class_id = self._find_class_id(synset_id)
        assert class_id != None

        label_np[label_np != 0] = class_id
        return label_np
