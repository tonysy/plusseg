from plusseg.utils.env import setup_environment

import argparse
import os

import torch
from plusseg.config import cfg
from plusseg.data import make_data_loader
# from plusseg.engine

from plusseg.modeling.segmentor import build_segmentation_model

from plusseg.utils.checkpoint import SegmentationCheckpointer

from plusseg.utils.collect_env import collect_env_info
from plusseg.utils.comm import synchronize, get_rank
from plusseg.utils.logger import setup_logger
from plusseg.utils.miscellaneous import mkdir
from plusseg.modeling.evaluator import MultiScaleEvaluator

try:
    from apex import amp
    from apex.parallel import SyncBatchNorm
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")

def main():
    parser = argparse.ArgumentParser("PyTorch Semantic Segmentation Inference")
    parser.add_argument(
        "--config-file",
        default="experiments/configs/fcn_res50_pcontext.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config option using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("plusseg", save_dir, get_rank())
    logger.info("Using {} GPUs".fromat(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_segmentation_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = SegmentationCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_dir)
            output_folders[idx] = output_folder
    data_loader_vals = make_data_loader(cfg, is_train=False,is_distributed=False)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loader_vals
    ):
        evaluator = MultiScaleEvaluator(
            model, cfg, devce_ids=range(num_gpus)
        )
        evaluator.eval()
        metric = seg_metric(cfg.DATASETS.NUM_CLASS)
        tqdm_bar = tqdm_bar(data_loader_val)
        for i, (image, target) in enumerate(tqdm_bar):
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(target, predicts)
                pix_acc, mIoU, IoU = metric.get()
                tqdm_bar.set_description(
                    'pix_acc: %.4f, mIoU: %.4f' % (pix_acc, mIoU)
                )
                