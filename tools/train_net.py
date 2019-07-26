r"""
Main training script for PyTorch
"""
from plusseg.utils.env import setup_environment

import argparse
import os

import torch

from tensorboardX import SummaryWriter

from plusseg.config import cfg
from plusseg.data import make_data_loader
from plusseg.solver import make_lr_scheduler
from plusseg.solver import make_optimizer

from plusseg.engine.inference import inference
from plusseg.engine.trainer import do_train

from plusseg.modeling.segmentor import build_segmentation_model

from plusseg.utils.checkpoint import SegmentationCheckpointer
from plusseg.utils.collect_env import collect_env_info
from plusseg.utils.comm import synchronize, get_rank
from plusseg.utils.imports import import_file
from plusseg.utils.logger import setup_logger
from plusseg.utils.miscellaneous import mkdir

try:
    from apex import amp
    from apex.parallel import SyncBatchNorm
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


