import math
import torch

class LRScheduler(object):
    """
    Learning Rate Scheduler

    - Step mode: ``lr = base_lr * 0.1 ^{floor(epoch - 1 / lr_step)}``
    
    - Cosine mode: ``lr = base_lr * 0.5 * (1 + cos(iter/max_iter))``

    - Poly mode: ``lr = base_lr * (1 - iter/maxiter) ^ 0.9``

    Args:
        cfg:  :config dict
    """
    def __init__(self, cfg, iters_per_epoch=0):
        self.mode = cfg.SOLVER.LR_SCHEDULER.MODE # step, cosine, poly
        self.base_lr = cfg.SOLVER.BASE_LR
        
        if self.mode == 'step':
            self.lr_decay_step = cfg.SOLVER.LR_SCHEDULER.LR_DECAY_STEP
            assert self.lr_decay_step
        else:
            self.lr_decay_step = 0

        self.iters_per_epoch = iters_per_epoch
        self.total_iters = cfg.SOLVER.EPOCHS * iters_per_epoch

        # self.epoch = - 1
        self.warmup_iters = cfg.SOLVER.LR_SCHEDULER.WARMUP_EPOCHS * iters_per_epoch

    def __call__(self, optimizer, iter_id, epoch):
        current_iters = epoch * self.iters_per_epoch + iter_id 

        if self.mode == 'cos':
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * current_iters / self.total_iters * math.pi))
        elif self.mode == 'poly':
            lr = self.base_lr * pow((1 - 1.0 * current_iters / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** epoch // self.lr_decay_step)
        else:
            raise NotImplementedError
        # warmup lr 
        if self.warmup_iters > 0 and current_iters < self.warmup_iters:
            lr = lr * 1.0 * current_iters / self.warmup_iters
        # if epoch > self.epoch:
        #     self.ep

        assert lr >= 0
        self.adjust_lr(optimizer, lr)
    
    def adjust_lr(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr 
        else:
            # import pdb; pdb.set_trace()
            # print('Todo: Check which case use this kind of lr_scheduler')
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr #* 10