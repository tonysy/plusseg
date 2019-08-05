import os
import numpy as np 

import torch

from PIL import Image
from tqdm import trange

from plusseg.data.base_dataset import BaseDataset

class PascalContextSegDataset(BaseDataset):
    BASE_DIR = 'VOCdevkit/VOC2010'
    NUM_CLASS = 59
    def __init__(self, root, split='train', mode=None, transform=None, target_transform=None, **kwargs):
        super(PascalContextSegDataset, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )

        from detail import Detail 
        root = os.path.join(root, self.BASE_DIR)
        ann_file = os.path.join(root, 'trainval_merged.json')
        img_dir = os.path.join(root, 'JPEGImages')

        # Trainig mode
        self.detail = Detail(ann_file, img_dir, split)
        self.transform = transform
        self.target_transform = self.target_transform
        self.ids = self.detail.getImgs()

        # Generated masks
        self.label_mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115
        ]))

        self.keys = np.array(range(len(self.label_mapping))).astype('uint8')
        mask_file = os.path.join(root, self.split+'.pth')
        print('Pascal Context Dataset, Mask File:{}'.format(mask_file))
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self.preprocess_mask(mask_file)
    
    def class2index(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self.label_mapping)
        index = np.digitize(mask.ravel(), self.label_mapping, right=True)
        return self.keys[index].reshape(mask.shape)

    def preprocess_mask(self, mask_file):
        """Generate mask files for pascal context dataset

        Args: 
            mask_file: (str) file path
        """
        masks = {}
        tbar = trange(len(self.ids))
        print('Preprocess the segmentation masks files for the first time running, this will take a while')
        for i in tbar:
            img_id = self.ids[i]
            mask = Image.fromarray(
                            self.class2index(
                                self.detail.getMask(img_id)
                                )
                            )
            masks[img_id['image_id']] = mask
            tbar.set_description("Preprocess {}".format(img_id['image_id']))
        
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.detail.img_folder, path)).convert('RGB')

        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(path)
        
        # Convert the mask to 60 categories
        mask = self.masks[iid]

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self.sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self.val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self.mask_transform(mask)
        
        # General Resize, Normalize and ToTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.ids)

    @property
    def pred_offset(self):
        return 1