import datetime
import logging
import time

import torch
import torch.distributed as dist

from plusseg.utils.comm import get_world_size, get_rank
from plusseg.utils.metric_logger import MetricLogger

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dict from all processes so that process with rank 0 has the averaged results.
    Returns a dict with the same fields as loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_loss = {k: v for k, v in zip(loss_names, all_losses)}
    
    return reduced_loss

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    total_iters,
    writer=None,
):
    logger = logging.getLogger('plusseg.trainer')
    logger.info(str(model))
    logger.info('Start Training')

    meters = MetricLogger(delimier=" ")
    max_epochs = arguments['epochs']
    start_epochs = arguments['max_epochs']
    iters_per_epoch = len(data_loader)
    logger_interval = arguments['logger_interval']
    
    model.train()
    start_training_time = time.time()
    end = time.time()

    for epoch_idx in range(start_epochs, start_epochs+max_epochs):
        
        for iteration, (images, targets) in enumerate(data_loader):
            data_time = time.time() - end 
            scheduler(optimizer, iteration, epoch_idx)

            images = images.to(device)
            targets = targets.to(device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduced losses over all GPUs for logging purpose
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()

            # Note: if mixed precision is not used, this ends up
            # doing nothing, otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

            optimizer.backward()
            batch_time = time.time() - end

            meters.update(time=batch_time, data=data_time)

            eta_iterations = total_iters - epoch_idx * iters_per_epoch - iteration
            eta_seconds = meters.time.global_avg * eta_iterations
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % logger_interval == 0 or iteration == iters_per_epoch:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta:{eta}",
                            "iter: {iter}/{max_iter}",
                            "epochs: {epoch_idx}/{max_epochs}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        max_iter=iters_per_epoch,
                        epoch_idx=epoch_idx,
                        max_epochs=max_epochs,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated()/1024.0/1024.0,
                    )
                )

        checkpointer.save('model_{:04d}'.format(epoch_idx), **arguments)
    
    checkpointer.save('model_final', **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (iters_per_epoch * max_epochs)
        )
    )