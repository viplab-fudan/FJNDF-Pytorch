import fnmatch
import re
import yaml
import os

from pyjnd.archs.inference_arch import InferenceModel
from pyjnd.default_model_configs import DEFAULT_CONFIGS
from pyjnd.utils import get_root_logger

def create_metric(metric_name, as_loss=False, device=None, **kwargs):
    assert metric_name in DEFAULT_CONFIGS.keys(), f'Metric {metric_name} not implemented yet.' 
    metric = InferenceModel(metric_name, as_loss=as_loss, device=device, **kwargs)
    logger = get_root_logger()
    logger.info(f'Metric [{metric.net.__class__.__name__}] is created.')
    return metric

def get_model(cfg, device=None):
    model = create_metric(cfg['name'], device=device, precision=cfg['precision'])
    if 'pretrained_model_path' in cfg:
        model.load_weights(cfg['pretrained_model_path'])
    model.net.eval()
    return model

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def get_dataset_info(dataset_name=None):
    dataset_info = yaml.safe_load(open(f'{os.path.dirname(os.path.abspath(__file__))}/default_dataset_configs.yml', 'r'))
    if dataset_name == None:
        return dataset_info
    else:
        assert dataset_name in dataset_info.keys(), f'Dataset {dataset_name} not implemented yet.'
        return dataset_info[dataset_name]
