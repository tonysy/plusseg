# Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.

import os

from yacs.config import CfgNode as CN

_C = CN()


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Base size of the image during training
_C.INPUT.IMAGE_SIZE = (480, 480)  # (height, width)
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
# _C.INPUT.TO_BGR255 = True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# -----------------------------------------------------------------------------
# Encoder Structure
# -----------------------------------------------------------------------------
# Encoder of the model
_C.MODEL.ENCODER = CN()
# Backbone for encoder structure
_C.MODEL.ENCODER.BACKBONE = CN()
# Dilate Convolution used for the last 3 stages
_C.MODEL.ENCODER.BACKBONE.DILATION = False


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
