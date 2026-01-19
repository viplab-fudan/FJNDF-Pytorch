"""
This file defines the base model for spatial-domain JND (Just Noticeable Difference) modeling.

It calculates a JND threshold by fusing different perceptual effects (e.g., luminance adaptation, contrast masking)
and provides several methods to inject or reduce noise in the spatial domain based on this threshold.

Five injection strategies are supported, controlled by the `target` parameter:
- target=0: Random additive noise injection.
- target=1: Block-mean smoothing.
- target=2: JND-bounded Gaussian smoothing.
- target=3: JND-guided adaptive Wiener filtering.
- target=4: JND-bounded Guided filtering (edge-preserving).
"""
import os
import yaml
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict

from pyjnd.utils.registry import MODEL_REGISTRY

from pyjnd.models.base_model import get_spatial_effect_registry

from pyjnd.archs.jnd_util import fuse_jnd_maps
from pyjnd.utils.img_util import inject_jnd


@MODEL_REGISTRY.register()
class SpatialJNDModel(nn.Module):
    """
    Base model for spatial-domain JND, with a structure and logic
    almost identical to the `SpatialJNDModel`.
    """
    def __init__(self,
                 effect_yml_path: str,
                 precision=None):
        super(SpatialJNDModel, self).__init__()

        if not effect_yml_path or not os.path.isfile(effect_yml_path):
            raise FileNotFoundError(
                f"SpatialJNDModel: config file not found: {effect_yml_path}"
            )

        with open(effect_yml_path, 'r', encoding='utf-8') as f:
            opt = yaml.safe_load(f)

        cfg = opt

        if 'effect' not in cfg:
            raise ValueError("Config file missing top-level 'effect' key")
        effects_cfg = cfg['effect']
        if not isinstance(effects_cfg, list):
            raise ValueError("'effect' must be a list in config file")

        if 'fusion' not in cfg:
            raise ValueError("Config file missing top-level 'fusion' key")
        fusion_cfg = cfg.get('fusion', {})
        self.fusion_method = fusion_cfg.get('method')

        registry = get_spatial_effect_registry()

        self.branches = []
        for idx, eff in enumerate(effects_cfg):
            # Basic field validation for each effect entry
            if 'type' not in eff or 'model' not in eff:
                raise ValueError(
                    f"Effect entry #{idx} must contain 'type' and 'model': {eff}"
                )
            eff_type = eff['type']
            model_key = eff['model']
            weight = eff.get('weight')
            params = eff.get('params', {})

            if eff_type not in registry:
                supported = ','.join(registry.keys())
                raise ValueError(
                    f"Unsupported effect type '{eff_type}' in entry #{idx}. "
                    f"Supported types: [{supported}]"
                )
            model_dict = registry[eff_type]

            if model_key not in model_dict:
                supported = ','.join(model_dict.keys())
                raise ValueError(
                    f"Unsupported model '{model_key}' for type '{eff_type}' in entry #{idx}. "
                    f"Supported models: [{supported}]"
                )
            model_cls = model_dict[model_key]

            inst = model_cls(**params)
            self.branches.append((eff_type, inst, weight))

        self.channels = cfg['channel']
        self.seed = cfg['seed']

    def compute(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Computes the final JND map.
        1) Calls the predict() method of each branch to get individual JND maps.
        2) Collects all JND maps and their corresponding weights.
        3) Fuses them using fuse_jnd_maps.
        4) Outputs the fused JND map.
        """
        maps: Dict[str, torch.Tensor] = {}
        weights: Dict[str, float] = {}
        for eff_type, inst, w in self.branches:
            Ji = inst.predict(yuv)   # (H, W)
            maps[eff_type] = Ji
            weights[eff_type] = w

        jnd_map = fuse_jnd_maps(maps, weights, self.fusion_method)

        return jnd_map.expand_as(yuv)

    def inject(
        self,
        yuv: torch.Tensor,
        jnd_map: torch.Tensor,
        channels: str = 'Y',
        max_val: float = 255.0,
        seed: int = None,
        target: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Injects or reduces noise in the spatial domain based on the JND threshold.

        target = 0: Randomly add/subtract T_JND.
        target = 1: Block-mean smoothing.
        target = 2: JND-bounded Gaussian smoothing.
        target = 3: JND-guided adaptive Wiener filtering.
        target = 4: JND-bounded Guided filtering (edge-preserving).
        """

        channels = self.channels
        seed = self.seed

        return inject_jnd(
            original=yuv,
            jnd_map=jnd_map,
            channels=channels,
            max_val=max_val,
            seed=seed,
            target=target
        )

    def forward(self, yuv: torch.Tensor,  **kwargs) -> torch.Tensor:
        """
        Default flow: compute JND -> inject JND -> return the injected image.
        """
        freq_spa_map = self.compute(yuv)

        channels = self.channels
        seed = self.seed
        target = kwargs.pop('target', 0)
        max_val = 255.0

        contaminated = self.inject(
            yuv=yuv,
            jnd_map=freq_spa_map,
            channels=channels,
            max_val=max_val,
            seed=seed,
            target=target
        )

        return contaminated