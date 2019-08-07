# Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.

import os
import torchvision.transforms as transform

class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        # citys
        "cityscape_train":{

        },
        "cityscape_val":{

        },
        "cityscape_test":{

        },
        
        # ade20k 
        "ade20k_train":{

        },
        "ade20k_val":{

        },
        "ade20k_test":{

        },

        # pascal voc
        "pascal_voc":{

        },
        "pascal_voc":{

        },
        "pascal_voc":{

        },
        
        # pascal context
        "pascal_context_train":{
            "img_dir": "PascalDataset/VOCdevkit/VOC2010/JPEGImages",
            "ann_file": "PascalDataset/VOCdevkit/VOC2010/trainval_merged.json",
            "mask_file":  "PascalDataset/VOCdevkit/VOC2010/train.pth",
            "split":'train',
            "mode": 'train',
            'transform': transform.Compose([
                            transform.ToTensor(),
                            transform.Normalize([.485, .456, .406], [.229, .224, .225])]
                        )
        },
        "pascal_context_val":{
            "img_dir": "PascalDataset/VOCdevkit/VOC2010/JPEGImages",
            "ann_file": "PascalDataset/VOCdevkit/VOC2010/trainval_merged.json",
            "mask_file":  "PascalDataset/VOCdevkit/VOC2010/val.pth",
            "split":'val',
            "mode": 'val',
            'transform': transform.Compose([
                            transform.ToTensor(),
                            transform.Normalize([.485, .456, .406], [.229, .224, .225])]
                        )
        },
        "pascal_context_test":{
            "img_dir": "PascalDataset/VOCdevkit/VOC2010/JPEGImages",
            "ann_file": "PascalDataset/VOCdevkit/VOC2010/trainval_merged.json",
            "mask_file":  "PascalDataset/VOCdevkit/VOC2010/test.pth",
            "split":'test',
            "mode": 'test',
            'transform': transform.Compose([
                            transform.ToTensor(),
                            transform.Normalize([.485, .456, .406], [.229, .224, .225])]
                        )
        },

        # camvid
        "camvid":{

        },
    }

    @staticmethod
    def get(name):
        if 'pascal_context' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                img_dir=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                mask_file=os.path.join(data_dir, attrs["mask_file"]),
                split=attrs["split"],
                mode=attrs["mode"],
                transform=attrs["transform"],
            )
            return dict(
                factory="PascalContextSegDataset",
                args=args,
            )
        raise RuntimeError('Dataset not avaliable: {}'.format(name))


class ModelCatalog(object):
    # C2_SEGMENTATION_URL = "plus.sist.shanghaitech.edu.cn/files/plusseg/models"
    C2_SEGMENTATION_URL = "https://dl.fbaipublicfiles.com/detectron"

    # C2_IMAGENET_MODELS = {
    #     "MSRA/R-50": "",
    #     "MSRA/R-101": "",
    #     "MSRA/R-152": "",
    # }
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
    }



    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.C2_SEGMENTATION_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])

        return url