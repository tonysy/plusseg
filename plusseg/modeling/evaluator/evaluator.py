import math
import numpy as np

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn import functional as F

class MultiScaleEvaluator(DataParallel):
    def __init__(self,model, cfg, device_ids=None):
        super(MultiScaleEvaluator, self).__init__(model, device_ids)
        self.num_class = cfg.DATASETS.NUM_CLASS
        self.flip = cfg.EVALUATOR.FLIP
        self.scales = cfg.EVALUATOR.SCALES
        self.base_size = cfg.DATASETS.BASE_SIZE
        self.crop_size = cfg.DATASETS.CROP_SIZE
        self.up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    def parallel_forward(self, inputs,**kwargs):
        inputs = [(input.unsqueeze(0).cuda(device),) for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = []

        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs)-len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs)-len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        return outputs

    def forward(self, image):
        B, _, H, W = image.size()

        stride_reate = 2.0/3.0
        stride = int(self.crop_size * stride_reate) # 320?

        with torch.cuda.device_of(image):
            scores = image.new().resize_(B, self.num_class,H,W).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            
            if H > W:
                height = long_size
                width = int(1.0 * W * long_size / H + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * H * long_size / W + 0.5)
                short_size = height
            
            # resize the image to current size
            cur_img = self.resize_image(image, height, width, self.up_kwargs)
            if long_size <= self.crop_size:
                pad_img = self.pad_image(cur_img, self.crop_size)
                outputs = self.flip_inference(pad_image, self.flip)
                outputs = self.crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < self.crop_size:
                    pad_img = self.pad_image(cur_img, self.crop_size)
                else:
                    pad_img = cur_img
                _, _, padded_h, padded_w = pad_image.size()

                assert(padded_h >= height and padded_w >= width)

                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (pad_image - crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pad_image - crop_size)/stride)) + 1

                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(B, self.num_class, padded_h, padded_w).zero_().cuda()
                    count_norm = image.new().resize_(B, 1, padded_h, padded_w).zero_().cuda()
                
                # Grid Evaluation
                for idx_h in range(h_grids):
                    for idx_w in range(w_grids):
                        h_0 = idx_h * stride
                        w_0 = idx_w * stride
                        h_1 = min(h_0 + self.crop_size, padded_h)
                        w_1 = min(w_0 + self.crop_size, padded_w)

                        crop_img = self.crop_image(pad_img, h_0, h_1, w_0, w_1)
                        # pad if neede
                        pad_crop_img = self.pad_image(crop_img, self.crop_size)
                        output = self.flip_inference(pad_crop_img, self.flip)
                        outputs[:, :, h_0:h_1, w_0:w_1] += self.crop_image(output, 0, h_1-h_0, 0, w_1-w_0)
                        count_norm[:,:,h_0:h_1,w_0:w_1] += 1
                assert ((count_norm ==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:, :height, :width]

            score = self.resize_image(outputs, H, W, self.up_kwargs)
            scores += score
        
        return scores

    def flip_inference(self, img, flip):
        output = self.model(img)
        if flip:
            f_img = self.flip_image(img)
            f_output = self.model(f_img)
            output += self.flip_image(f_output)
        return output.exp()

    def resize_image(self, img, h, w, **up_kwargs):
        return F.interpolate(img,size=(h, w), **up_kwargs)

    def pad_image(self, img, crop_size):
        B, C, H, W = img.size()
        assert(C==3)
        pad_h = crop_size - H if H < crop_size else 0
        pad_w = crop_size - W if W < crop_size else 0
        
        pad_values = -np.array(self.mean) / np.array(self.std)
        img_pad = img.new().resize_(B, C, H+pad_h, H+pad_w)

        for i in range(C):
            img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, pad_w, 0, pad_h), value=pad_values[i])
        
        assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)

        return img_pad

    def flip_image(img):
        assert(img.dim() == 4)
        with torch.cuda.device_of(img):
            idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
        
        return img.index_select(3, idx)

    def crop_image(img, h0, h1, w0, w1):
        return img[:,:,h0:h1, w0:w1]
