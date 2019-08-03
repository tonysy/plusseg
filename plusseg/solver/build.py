import torch
from .lr_scheduer import LRScheduler

def make_optimizer(cfg, model, lr_factor_dict):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if 'bias' in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight = cfg.SOLVER.WEIGHT_DECAY_BIAS
        
        params += [{"params":[value], "lr":lr, "weight_decay":weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer

def make_lr_scheduler(cfg, iters_per_epoch):
    return LRScheduler(cfg, iters_per_epoch)