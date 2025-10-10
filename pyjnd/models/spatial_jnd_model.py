# spatial_jnd.py
import os
import yaml
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union
from collections import OrderedDict

from pyjnd.utils.registry import MODEL_REGISTRY

import pyjnd.models
from pyjnd.models.base_model import get_spatial_effect_registry

from pyjnd.archs.jnd_util import fuse_jnd_maps
from pyjnd.utils.img_util import inject_jnd

from typing import Dict

@MODEL_REGISTRY.register()
class SpatialJNDModel(nn.Module):
    """
    Spatial-domain JND model. When constructing, only need to provide YAML file path, internally will automatically:
      1) Read top-level 'effect' list
      2) Check 'type' and 'model' for each entry
      3) Get corresponding class from _SPATIAL_EFFECT_REGISTRIES and instantiate
      4) Weight by 'weight', generate branch list

    Example spatial_jnd.yml:

    name: spatial_jnd

    effect:
      - type: LA
        model: davis2006
        weight: 1.0
        params:
          alpha: 0.5
      - type: CM
        model: preisach2009
        weight: 0.8
        params: {}
      - type: PM
        model: preisach2009
        weight: 0.8
        params: {}
    """
    def __init__(self,
                 effect_yml_path: str):
        super(SpatialJNDModel, self).__init__()
        # 1) Get opt configuration
        if not effect_yml_path or not os.path.isfile(effect_yml_path):
            raise FileNotFoundError(f"SpatialJNDModel: Configuration file does not exist:{effect_yml_path}")
        
        with open(effect_yml_path, 'r', encoding='utf-8') as f:
            opt = yaml.safe_load(f)
        
        cfg = opt

        # Read fusion configuration
        # 2) Ensure there are effect list and fusion list, get global registry
        if 'effect' not in cfg:
            raise ValueError(f"Config file missing top-level 'effect' key")
        effects_cfg = cfg['effect']
        if not isinstance(effects_cfg, list):
            raise ValueError(f"'effect' must be a list in Config file")

        if 'fusion' not in cfg:
            raise ValueError(f"Config file missing top-level 'fusion' key")
        fusion_cfg = cfg.get('fusion', {})
        self.fusion_method = fusion_cfg.get('method')

        registry = get_spatial_effect_registry()

        # 3) Iterate through each effect entry
        self.branches = []
        for idx, eff in enumerate(effects_cfg):
            # Basic field check
            if 'type' not in eff or 'model' not in eff:
                raise ValueError(
                    f"Effect entry #{idx} must contain 'type' and 'model': {eff}"
                )
            eff_type  = eff['type']
            model_key = eff['model']
            weight    = eff.get('weight')
            params    = eff.get('params', {})

            # 4) Find corresponding model dictionary according to type
            if eff_type not in registry:
                supported = ','.join(registry.keys())
                raise ValueError(
                    f"Unsupported effect type '{eff_type}' in entry #{idx}. "
                    f"Supported types: [{supported}]"
                )
            model_dict = registry[eff_type]

            # 5) Find specific class according to model
            if model_key not in model_dict:
                supported = ','.join(model_dict.keys())
                raise ValueError(
                    f"Unsupported model '{model_key}' for type '{eff_type}' in entry #{idx}. "
                    f"Supported models: [{supported}]"
                )
            model_cls = model_dict[model_key]

            # 6) Instantiate and add to branch list
            inst = model_cls(**params)
            self.branches.append((eff_type, inst, weight))

        # Load other parameter settings
        self.channels = cfg['channel']
        self.seed = cfg['seed']
        self.target = cfg['target']
        self.block_sz = cfg['block_sz']

    def compute(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        1) Call predict() of each branch to get J_i (C,H,W)
        2) Collect maps and weights
        3) Call fuse_jnd_maps
        4) Output (C,H,W)
        """
        maps : Dict[str, torch.Tensor] = {}
        weights : Dict[str, float] = {}
        for eff_type, inst, w in self.branches:
            Ji = inst.predict(yuv)   # (H,W)
            maps[eff_type] = Ji
            weights[eff_type] = w

        # Execute fusion
        jnd_map = fuse_jnd_maps(maps, weights, self.fusion_method)

        return jnd_map.expand_as(yuv)

    def inject(self, yuv: torch.Tensor, jnd_map: torch.Tensor,
               channels: str = 'Y',
               max_val: float = 255.0,
               seed: int = None,
               target: Optional[int] = 0,
               block_sz: Optional[int] = 8) -> torch.Tensor:
        """
        Call generic inject_jnd to inject jnd_map into yuv.
        """

        channels = self.channels
        seed     = self.seed
        target   = self.target
        block_sz = self.block_sz

        return inject_jnd(
            original=yuv,
            jnd_map=jnd_map,
            channels=channels,
            max_val=max_val,
            seed=seed,
            target=target,
            block_sz=block_sz
        )
    
    def forward(self,
                yuv: torch.Tensor
               ) -> torch.Tensor:
        """
        Standard PyTorch forward: first compute JND, then inject noisy YUV, return (C,H,W).
        Args:
          yuv      : Input YUV Tensor, shape = (C,H,W)
        Returns:
          contaminated: torch.Tensor, shape = (C,H,W)
        """

        channels = self.channels
        seed     = self.seed
        target   = self.target
        block_sz = self.block_sz
        max_val  = 255.0

        # 1) Calculate JND map
        jnd_map = self.compute(yuv)
        # 2) Inject JND noise
        contaminated = self.inject(
            yuv=yuv,
            jnd_map=jnd_map,
            channels=channels,
            max_val=max_val,
            seed=seed,
            target=target,
            block_sz=block_sz
        )

        max_jnd = jnd_map.max()

        if max_jnd > 0:
            jnd_map = (jnd_map / max_jnd) * max_val

        return contaminated