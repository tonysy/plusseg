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



def train(cfg, local_rank, distributed):
    model = build_segmentation_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)



def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--no-validate",
        dest='no_validate',
        help="Dont validate after each epoch",
        action="store_ture",
    )

    parser.add_argument(
        "opts",
        hlep='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    
    logger = setup_logger("plusseg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    if cfg.COLLECT_ENV_INFO:
        logger.info("Collecting env info(might take some time)")
        logger.info("\n"+collect_env_info())
    
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" +cfg.read()
        logger.info(config_str)

    logger.info("Running with config: \n{}".format(cfg))
    model = train(cfg, args.local_rank, args.distributed, args.no_validate)



if __name__ == "__main__":
    main()