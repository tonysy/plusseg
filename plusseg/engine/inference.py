import logging
import time
import os

import torch
from tqdm import tqdm

from plusseg.config import cfg
from plusseg.data.datasets.evaluation import evaluate

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

# def compute_on_dataset(model, data_loader, device, timer=None):
#     model.eval()
#     results_dict = {}
#     cpu_device = torch.device("cpu")
#     for _, batch in enumerate(tqdm(data_loader)):
#         images, targets = batch
#         with torch.no_grad():
#             if timer:
#                 timer.tic()
#             output = model(images.to(device))
#             if timer:
#                 torch.cuda.synchronize()
#                 timer.toc()
#             output = [o.to(cpu_device) for o in output]
        

def inference(
    model,
    data_loader,
    dataset_name,
    device="cuda",
    expected_results=(),
    output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("plusseg.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset {} images".format(
        dataset_name, len(dataset)
    ))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} /({}s/img per device, on {} devices)".format(total_time_str, total_time * num_devices / len(dataset), num_devices)
    )

    total_inference_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time:{} ({} s/img per device, on {} devices)".format(total_inference_time,
         inference_timer.total_time  * num_devices / len(dataset),
         num_devices)
    )

    if not is_main_process():
        return 

    # if output_folder:
        # torch.save(predictions, os.)
