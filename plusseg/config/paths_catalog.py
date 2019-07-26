# Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.

import os

class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        "cityscape":{

        },
        "ade20k":{

        },
        "pascalvoc":{

        },
        "pascalcontext":{

        },
        "camvid":{

        },
    }

    @staticmethod
    def get(name):
        pass

class ModelCatalog(object):
    C2_SEGMENTATION_URL = "plus.sist.shanghaitech.edu.cn/files/plusseg/models"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "",
        "MSRA/R-101": "",
        "MSRA/R-152": "",
    }


    @staticmethod
    def get(name):
        pass

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.C2_SEGMENTATION_URL
        name = name[len("ImageNetPretrained/")]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])

        return url