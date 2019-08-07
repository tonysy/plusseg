# # Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.
# import bisect
# import copy
import logging

# import torch.utils.data
from plusseg.utils.comm import get_world_size
from plusseg.utils.imports import import_file

# from . import datasets as D
from . import samplers

# from .transforms import build_transforms

from .base_dataset import BaseDataset
from .datasets import PascalContextSegDataset

datasets = {
    "coco": None,
    "pcontext": PascalContextSegDataset
}

def build_dataset(cfg):
    """
    Arguments:
        cfg: (dict) configureation parameters
    """
    dataset_name = cfg.DATASET.NAME
    return datasets[dataset_name.lower()](cfg)



def make_data_loader(cfg, is_train=True, is_distributed=False):
    num_gpus  = get_world_size()
    if is_train:
        imgs_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            imgs_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            imgs_per_batch, num_gpus)

        imgs_per_gpu = imgs_per_batch // num_gpus
        shuffle = True

        # num_iters = cfg.SOLVER.MAX_ITER
    
    else:
        imgs_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            imgs_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            imgs_per_batch, num_gpus)
        
        imgs_per_gpu = imgs_per_batch // num_gpus
        shuffle = False if not is_distributed else True
    
    if imgs_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

        paths_catalog = import_file(
            "plusseg.config.paths_catalog",
            cfg.PATHS_CATALOG, True
        )

        DatasetCatalog = paths_catalog.DatasetCatalog
        dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST 
        # import pdb; pdb.set_trace()
        # transforms = None if not is_train else build_transforms(cfg, is_train)
        # datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)
        datasets = build_dataset(dataset_list, DatasetCatalog, is_train)
        data_loaders = []
        for dataset in datasets:
            sampler = make_data_sampler(dataset, shuffle, is_distributed)
            num_workers = cfg.DATALOADER.NUM_WORKERS 
            data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,     
                sampler=sampler
            )
            data_loaders.append(data_loader)
        
        if is_train:
           assert len(data_loaders) == 1
           return data_loaders[0]

        return data_loaders 

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

# def make_batch_data_sampler(dataset, sampler, images_per_batch):
    # batch_sampler = torch.utils.data.sampler.BatchSampler(
        # sampler, images_per_batch, drop_last=False
    # )