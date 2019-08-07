import logging
import os

import torch

from plusseg.utils.model_serialization import  load_state_dict
from plusseg.utils.c2_model_loading import load_c2_format
from plusseg.utils.imports import import_file
from plusseg.utils.model_zoo import cache_url

class Checkpointer(object):
    def __init__(self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,):
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        
        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        # if self.scheduler is not None:
        #     data['scheduler'] = self.scheduler.state_dict()

        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.path".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            f = self.get_checkpoint_file()
        if not f:
            self.logger.info("No checkpoint found. Initializng model from scratch")
            return {}
        
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        # if "scheduler" in checkpoint and self.scheduler:
        #     self.logger.info("Loading scheduler from {}".format(f))
        #     self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint 

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)
    
    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        try:
            with open(save_file, 'r') as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
                last_saved = ""
            
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))
    
    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop('model'))

class SegmentationCheckpointer(Checkpointer):
    def __init__(self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir='',
        save_to_disk=None,
        logger=None,):
        super(SegmentationCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "plusseg.config.paths_catalog",
                self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} Points to {}".format(f, catalog_f))
            f = catalog_f
        
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f

        # convert caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        
        # load native detectron.pytorch checkpoint
        loaded = super(SegmentationCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
        