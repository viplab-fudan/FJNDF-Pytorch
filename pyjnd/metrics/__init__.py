import importlib

from copy import deepcopy
from os import path as osp

from pyjnd.utils.registry import METRIC_REGISTRY
from pyjnd.utils import scandir

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_metric.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'pyjnd.metrics.{file_name}') for file_name in model_filenames]

__all__ = ['calculate_metric']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(*data, **opt)
    return metric
