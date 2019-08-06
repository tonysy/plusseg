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
_C.MODEL.META_ARCHITECTURE = 'GeneralizeSegmentor'
_C.MODEL.BATCH_NORM = 'SyncBatchNorm' # nn.BatchNorm2d
_C.MODEL.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Encoder Structure
# -----------------------------------------------------------------------------
# Encoder of the model
_C.MODEL.ENCODER = CN()
# Backbone for encoder structure
_C.MODEL.ENCODER.BACKBONE = CN()
# Dilate Convolution used for the last 3 stages
_C.MODEL.ENCODER.BACKBONE.CONV_BODY = "R-50-C5"

_C.MODEL.ENCODER.BACKBONE.DILATION = False
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.ENCODER.BACKBONE.FREEZE_CONV_BODY_AT = -1 # -1 means no frozen


# ResNet structure
_C.MODEL.ENCODER.BACKBONE.RESNETS = CN()
_C.MODEL.ENCODER.BACKBONE.RESNETS.STRIDE_IN_1X1 = True

_C.MODEL.ENCODER.BACKBONE.RESNETS.STEM_FUNC = 'StemWithSyncBN'
_C.MODEL.ENCODER.BACKBONE.RESNETS.TRANS_FUNC = 'BottleneckWithSyncBN'
_C.MODEL.ENCODER.BACKBONE.RESNETS.STEM_OUT_CHANNELS = 64
_C.MODEL.ENCODER.BACKBONE.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.ENCODER.BACKBONE.RESNETS.NUM_GROUPS = 1
_C.MODEL.ENCODER.BACKBONE.RESNETS.WIDTH_PER_GROUP = 64

_C.MODEL.ENCODER.BACKBONE.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 8

_C.MODEL.ENCODER.BACKBONE.RESNETS.RES5_DILATION = 1

_C.MODEL.ENCODER.BACKBONE.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.ENCODER.BACKBONE.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.ENCODER.BACKBONE.RESNETS.DEFORMABLE_GROUPS = 1


# -----------------------------------------------------------------------------
# Decoder Structure
# -----------------------------------------------------------------------------
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NAME = 'FCN'
_C.MODEL.DECODER.AUX_FACTOR = 0.2

_C.MODEL.DECODER.FCN = CN()
_C.MODEL.DECODER.FCN.IN_CHANNEL = 2048
_C.MODEL.DECODER.FCN.CHANNEL_STRIDE = 4
_C.MODEL.DECODER.FCN.AUX_IN_CHANNEL = 1024
_C.MODEL.DECODER.FCN.DROPOUT = 0.1
_C.MODEL.DECODER.FCN.OUT_CHANNEL = 59

_C.MODEL.POSTPROCESSOR = CN()
_C.MODEL.POSTPROCESSOR.NAME = ''

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()




# -----------------------------------------------------------------------------
# Misc Options
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "."

_C.COLLECT_ENV_INFO = False
