import importlib
import json

from copy import deepcopy
from os import path as osp
from typing import Dict, Optional
from pathlib import Path

from pyjnd.utils import get_root_logger, scandir
from pyjnd.utils.registry import ARCH_REGISTRY
from pyjnd.default_model_configs import DEFAULT_CONFIGS

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_arch.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'pyjnd.archs.{file_name}') for file_name in model_filenames]


__all__ = ['build_model', 'list_models', 'get_dataset_info']

def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    model_name = opt['arch_type']
    assert model_name in DEFAULT_CONFIGS.keys(), f'Model {model_name} not implemented yet.' 
    model = ARCH_REGISTRY.get(opt['arch_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Arch [{model.__class__.__name__}] is created.')
    return model

def list_models(type=None, filter='', exclude_filters=''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
    Example:
        model_list('*ssim*') -- returns all models including 'ssim'
    """
    if type is None:
        all_models = DEFAULT_CONFIGS.keys()
    else:
        assert type in ['SpatialJNDModel', 'FrequencyJNDModel', 'TopDownJNDModel', 'GeneralLRJNDModel']
        f'Model Type only support [SpatialJNDModel, FrequencyJNDModel, TopDownJNDModel, GeneralLRJNDModel], but got {type}'
        all_models = [key for key in DEFAULT_CONFIGS.keys() if DEFAULT_CONFIGS[key]['type'] == type]

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    return list(sorted(models, key=_natural_key))


def get_dataset_info(dataset_name=None):
    dataset_info = yaml.safe_load(open(f'{osp.dirname(osp.abspath(__file__))}/default_dataset_configs.yml', 'r'))
    if dataset_name == None:
        return dataset_info
    else:
        assert dataset_name in dataset_info.keys(), f'Dataset {dataset_name} not implemented yet.'
        return dataset_info[dataset_name]