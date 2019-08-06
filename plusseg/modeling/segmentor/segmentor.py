from .generalized_segmentor import GeneralizeSegmentor 

_SEGMENTATION_META_ARCHITECTURE = {'GeneralizeSegmentor':GeneralizeSegmentor}

def build_segmentation_model(cfg):
    meta_arch = _SEGMENTATION_META_ARCHITECTURE[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)