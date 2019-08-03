from .generalized_segmentor import GeneralizeSegmentor 

_SEGMENTATION_META_ARCHITECTURES = {'GeneralizeSegmentor':GeneralizeSegmentor}

def build_segmentation_model(cfg):
    meta_arch = _SEGMENTATION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURES]
    return meta_arch(cfg)