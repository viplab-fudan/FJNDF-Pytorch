"""
This file defines the base model for frequency-domain JND (Just Noticeable Difference) modeling.

It calculates a JND threshold by fusing different perceptual effects (e.g., CSF, luminance adaptation)
and provides several methods to inject or reduce noise in the frequency domain based on this threshold.

Five injection strategies are supported, controlled by the `target` parameter:
- target=0: Random additive noise injection.
- target=1: Magnitude reduction (similar to JPEG quantization).
- target=2: Weighted magnitude reduction based on frequency.
- target=3: Adaptive Wiener filtering based on JND.
- target=4: JND smoothing based on block classification.
"""
import yaml
import math
import os

import torch
import torch.nn.functional as F
import torch_dct as tdct
from torch import nn

from typing import Dict, Optional

import pyjnd.models
from pyjnd.utils.registry import MODEL_REGISTRY
from pyjnd.utils import block_transform, block_idtransform
from pyjnd.models.base_model import get_frequency_effect_registry
from pyjnd.archs.jnd_util import fuse_jnd_maps

@MODEL_REGISTRY.register()
class FrequencyJNDModel(nn.Module):
    """
    Base model for frequency-domain JND, with a structure and logic
    almost identical to the `SpatialJNDModel`.
    """
    def __init__(self,
                 effect_yml_path: str,
                 precision=None):
        super(FrequencyJNDModel, self).__init__()
        # 1) Get and parse the configuration file
        if not effect_yml_path or not os.path.isfile(effect_yml_path):
            raise FileNotFoundError(f"FrequencyJNDModel: Configuration file not found: {effect_yml_path}")
        
        with open(effect_yml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # 2) Check config and get the global registry
        if 'effect' not in cfg:
            raise ValueError(f"Config file {effect_yml_path} is missing the 'effect' key")
        effects_cfg = cfg['effect']

        if 'fusion' not in cfg:
            raise ValueError(f"Config file {effect_yml_path} is missing the 'fusion' key")
        fusion_cfg = cfg.get('fusion', {})
        self.fusion_method = fusion_cfg.get('method')

        ttype = cfg.get('transform')
        self.transform_type = ttype

        # Select frequency transform functions based on config
        if ttype == 'dct2d_8x8':
            # 8x8 block-wise DCT-II / IDCT
            self.transform_fn     = lambda x: block_transform(x, tdct.dct_2d, 8)
            self.inverse_fn       = lambda X: block_idtransform(X, tdct.idct_2d, 8)
        elif ttype == 'dct2d':
            # Full image DCT-II / IDCT
            self.transform_fn     = tdct.dct_2d
            self.inverse_fn       = tdct.idct_2d
        else:
            raise ValueError(f"Unsupported transform type: {ttype}")

        registry = get_frequency_effect_registry()

        # 3) Iterate through each effect entry and instantiate the corresponding model
        self.branches = []
        for idx, eff in enumerate(effects_cfg):
            if 'type' not in eff or 'model' not in eff:
                raise ValueError(f"Effect entry #{idx} must contain 'type' and 'model' keys: {eff}")
            
            eff_type  = eff['type']
            model_key = eff['model']
            weight    = eff.get('weight')
            params    = eff.get('params', {})

            # Find and instantiate the model from the registry
            if eff_type not in registry:
                supported = ','.join(registry.keys())
                raise ValueError(f"Unsupported effect type '{eff_type}'. Supported types: [{supported}]")
            model_dict = registry[eff_type]

            if model_key not in model_dict:
                supported = ','.join(model_dict.keys())
                raise ValueError(f"Unsupported model '{model_key}' for type '{eff_type}'. Supported models: [{supported}]")
            model_cls = model_dict[model_key]

            inst = model_cls(**params)
            self.branches.append((eff_type, inst, weight))
        
        # Load other parameter settings
        self.channels = cfg['channel']
        self.seed = cfg['seed']
        self.target = cfg['target']
        self.block_sz = cfg['block_sz']

    def compute(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Computes the final JND map.
        1) Calls the predict() method of each branch to get individual JND maps.
        2) Collects all JND maps and their corresponding weights.
        3) Fuses them using fuse_jnd_maps.
        4) Outputs the fused JND map.
        """
        maps : Dict[str, torch.Tensor] = {}
        weights : Dict[str, float] = {}
        for eff_type, inst, w in self.branches:
            Ji = inst.predict(yuv)
            maps[eff_type] = Ji
            weights[eff_type] = w

        jnd_map = fuse_jnd_maps(maps, weights, self.fusion_method)

        return jnd_map.expand_as(yuv)
    
    def inject(
        self,
        yuv: torch.Tensor,
        freq_jnd_map: torch.Tensor,
        channels: str = 'YUV',
        seed: Optional[int] = None,
        target: Optional[int] = 0,
        block_sz: Optional[int] = 8,
    ) -> torch.Tensor:
        """
        Injects or reduces noise in the frequency domain based on the JND threshold,
        then transforms the result back to the pixel domain via IDCT.

        target = 0: Randomly add/subtract T_JND (original injection method).
                    Formula: C' = sign(C) * max(|C| + f * T_JND, 0), where f is a random {-1, +1}.
        target = 1: Reduce magnitude (JPEG optimization mode), keeping the sign.
                    Formula: C' = sign(C) * max(|C| - T_JND, 0).
        target = 2: Weighted magnitude reduction.
                    Formula: C' = sign(C) * sqrt(max(C^2 - p(u,v)*JND^2, 0)).
                    Coefficients are set to zero if |C| < JND.
        target = 3: Adaptive Gaussian filtering.
        target = 4: JND smoothing based on block classification.
        """
        C, H, W = yuv.shape
        out = yuv.clone()

        channels = self.channels
        seed     = self.seed
        target   = self.target
        block_sz = self.block_sz
        
        # Set random seed only for target=0 to ensure reproducibility
        if target == 0 and seed is not None:
            torch.manual_seed(seed)

        for c in range(C):
            # Determine if the current channel needs to be processed
            do_ch = (channels == 'YUV') or \
                    (channels == 'Y'  and c == 0) or \
                    (channels == 'UV' and c in (1, 2))
            if not do_ch:
                continue

            # Transform to the frequency domain
            coeff = self.transform_fn(yuv[c:c+1])

            # Get magnitude and original sign
            mag = coeff.abs()
            sign = coeff.sign()

            # Select processing method based on target
            thresh_c = freq_jnd_map[c].unsqueeze(0)

            if target == 0:
                # Randomly add/subtract T_JND
                noise_sign = (torch.randint(0, 2, coeff.shape, device=coeff.device, dtype=torch.int8) * 2 - 1).to(coeff.dtype)
                new_mag = (mag + noise_sign * thresh_c).clamp(min=0.0)
                coeff2 = sign * new_mag

            elif target == 1:
                # Reduce magnitude
                new_mag = (mag - thresh_c).clamp(min=0.0)
                coeff2 = sign * new_mag

            elif target == 2:
                # Weighted magnitude reduction
                p_block = torch.zeros((block_sz, block_sz), device=coeff.device, dtype=coeff.dtype)
                for i in range(block_sz):
                    for j in range(block_sz):
                        s = i + j
                        if s == 0: p_block[i, j] = 0.0
                        elif s <= 2: p_block[i, j] = 0.25
                        elif 3 <= s <= 6: p_block[i, j] = 0.75
                        else: p_block[i, j] = 1.0
                
                nH, nW = H // block_sz, W // block_sz
                p_map = p_block.repeat(nH, nW).unsqueeze(0)

                inner  = (mag.pow(2) - p_map * thresh_c.pow(2)).clamp(min=0.0)
                coeff2 = torch.sqrt(inner) * sign
                coeff2 = coeff2 * (mag >= thresh_c) # Set to zero if |C| < JND

            elif target == 3:
                # Adaptive Wiener filtering
                coeff2 = torch.empty_like(coeff)
                b = block_sz
                u_idx = torch.arange(b, device=coeff.device).view(b,1).expand(b,b).float()
                v_idx = torch.arange(b, device=coeff.device).view(1,b).expand(b,b).float()
                omega_map = torch.sqrt(u_idx**2 + v_idx**2)
                beta = math.log(2**0.5) / (4 * math.pi**2)
            
                H_blk, W_blk = H // b, W // b
                for by in range(H_blk):
                    for bx in range(W_blk):
                        y0, y1 = by*b, (by+1)*b
                        x0, x1 = bx*b, (bx+1)*b
            
                        blk = coeff[:, y0:y1, x0:x1].squeeze(0)
                        J_blk = thresh_c[:, y0:y1, x0:x1].squeeze(0)
                        Cabs = blk.abs() + 1e-6
            
                        # Calculate frequency-dependent sigma_d(u,v)
                        ratio = torch.clamp(J_blk / Cabs, max=0.9999)
                        denom = -torch.log(1 - ratio)
                        sigma_uv = omega_map * torch.sqrt(beta / denom)
            
                        # Get the most stringent sigma within the block
                        sigma_blk = sigma_uv.max()
            
                        # Generate gain G(u,v) and apply it
                        G = torch.exp(-beta * omega_map**2 / (sigma_blk**2))
                        coeff2[:, y0:y1, x0:x1] = (blk * G).unsqueeze(0)

            elif target == 4:
                # Block classification-based JND smoothing (8x8 DCT only)
                b = block_sz
                assert b == 8, "target=4 currently only supports 8x8 DCT"
                coeff2 = torch.empty_like(coeff)

                # Smoothing parameters
                SMOOTH_FACTOR_T = 0.000175
                DISORDER_RATIO_U = 10

                def classify_block(Fblk: torch.Tensor) -> str:
                    E_h = (Fblk[1:, 0] ** 2).sum()
                    E_v = (Fblk[0, 1:] ** 2).sum()
                    E_d = torch.diagonal(Fblk[1:, 1:]).pow(2).sum()
                    E_s = SMOOTH_FACTOR_T * (b - 1) * (Fblk[0, 0] ** 2)
                    
                    vals = torch.stack([E_s, E_h, E_v, E_d])
                    idx = int(torch.argmax(vals).item())
                    if idx == 0: return "smooth"
                    if torch.max(vals[1:]) / torch.min(vals[1:]) < DISORDER_RATIO_U: return "disorder"
                    return ["smooth", "hor", "ver", "dia"][idx]
                
                W_freq = torch.tensor([[0,2,2,6,6,6,6,6], 
                                       [2,2,6,8,8,8,8,8], 
                                       [2,6,8,8,8,8,8,8],
                                       [6,8,6,8,8,8,8,8], 
                                       [6,8,8,8,8,8,8,8], 
                                       [6,8,8,8,8,8,8,8],
                                       [6,8,8,8,8,8,8,8], 
                                       [6,8,8,8,8,8,8,8]], dtype=torch.float32, device=coeff.device)

                H_blk, W_blk = H // b, W // b
                for by in range(H_blk):
                    for bx in range(W_blk):
                        y0, y1 = by*b, (by+1)*b
                        x0, x1 = bx*b, (bx+1)*b

                        Fblk = coeff[:, y0:y1, x0:x1].squeeze(0)
                        J_blk = thresh_c[:, y0:y1, x0:x1].squeeze(0)

                        blk_type = classify_block(Fblk)
                        
                        # Generate directional weight alpha based on block type
                        u, v = torch.arange(b, device=coeff.device).view(b,1), torch.arange(b, device=coeff.device).view(1,b)
                        if blk_type == "ver": W_type = u.float()
                        elif blk_type == "hor": W_type = v.float()
                        elif blk_type == "dia": W_type = (u - v).abs().float()
                        elif blk_type == "disorder": W_type = torch.maximum(u, v).float()
                        else: # smooth
                            m = torch.maximum(u, v)
                            W_type = torch.ones_like(m, dtype=torch.float32) * 3
                            W_type[m <= 2] = 1.0
                            W_type[(m >= 3) & (m <= 5)] = 2.0
                        
                        alpha = (W_type * W_freq) / 64.0
                        alpha = alpha.clamp(0.0, 1.0)
                        
                        # Reduce magnitude based on the threshold
                        Cabs = Fblk.abs()
                        Ce = (Cabs.pow(2) - (alpha * J_blk).pow(2)).clamp(min=0.0).sqrt()
                        Fhat = Ce * Fblk.sign()
                        Fhat = Fhat * (Cabs >= (alpha * J_blk))
                        coeff2[:, y0:y1, x0:x1] = Fhat.unsqueeze(0)

            else:
                raise ValueError(f"Unsupported target type: {target}")
            
            # Inverse transform back to the pixel domain
            rec = self.inverse_fn(coeff2)[0]
            out[c] = rec
        return out

    def forward(
        self,
        yuv: torch.Tensor
    ) -> torch.Tensor:
        """
        Default flow: compute JND -> inject JND -> return the injected image.
        """
        freq_jnd_map = self.compute(yuv)

        channels = self.channels
        seed = self.seed
        target = self.target
        block_sz = self.block_sz

        contaminated = self.inject(yuv, freq_jnd_map, 
            channels=channels,
            seed=seed,
            target=target,
            block_sz=block_sz)

        return contaminated