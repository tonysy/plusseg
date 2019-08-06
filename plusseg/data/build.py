# Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from plusseg.utils.comm import get_world_size
from plusseg.utils.imports import import_file

from . import datasets as D
# from . import samplers

from .transforms import build_transforms

def build_dataset(dataset_list, transfroms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            ade20k_train, ade20k_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data['factory'])
        args = data["args"]

        args["transforms"] = transfroms

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets
    
    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


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

        num_iters = cfg.SOLVER.MAX_ITER
    
    else:
        imgs_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            imgs_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            imgs_per_batch, num_gpus)
        
        imgs_per_gpu = imgs_per_batch // num_gpus
        shuffle = False if not is_distributed else True
    
    if images_per_gpu > 1:
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

        path_catalog = import_file(
            "plusseg.config.paths_catalog",
            cfg.PATHS_CATALOG, True
        )

        DatasetCatalog = paths_catalog.DatasetCatalog
        dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST 

        transforms = None if not is_train else build_transfroms(cfg, is_train)
        datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

        data_loaders = []
        for dataset in datasets:
            num_workers = cfg.DATALOADER.NUM_WORKERS 
            data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,     
            )
            data_loaders.append(data_loader)
        
        if is_train:
           assert len(data_loaders) == 1
           return data_loaders[0]

        return data_loaders 