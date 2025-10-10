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
from pyjnd.utils import block_transform, block_idtransform, get_block_saliency_mask
from pyjnd.models.base_model import get_frequency_effect_registry
from pyjnd.archs.jnd_util import fuse_jnd_maps

@MODEL_REGISTRY.register()
class FrequencyJNDModel(nn.Module):
    """
    Frequcny Domain JND Base Model, Almost same as SaptialJNDModel
    """
    def __init__(self,
                 effect_yml_path: str,
                 precision=None):
        super(FrequencyJNDModel, self).__init__()
        # 1) Get opt configuration
        if not effect_yml_path or not os.path.isfile(effect_yml_path):
            raise FileNotFoundError(f"FrequencyJNDModel: Configuration file does not exist:{effect_yml_path}")
        
        with open(effect_yml_path, 'r', encoding='utf-8') as f:
            opt = yaml.safe_load(f)
        
        cfg = opt

        # Read fusion configuration
        # 2) Ensure there are effect list and fusion list, get global registry
        if 'effect' not in cfg:
            raise ValueError(f"Config file {config_path} missing top-level 'effect' key")
        effects_cfg = cfg['effect']
        if not isinstance(effects_cfg, list):
            raise ValueError(f"'effect' must be a list in {config_path}")

        if 'fusion' not in cfg:
            raise ValueError(f"Config file {config_path} missing top-level 'fusion' key")
        fusion_cfg = cfg.get('fusion', {})
        self.fusion_method = fusion_cfg.get('method')

        ttype = cfg.get('transform')
        self.transform_type = ttype

        if ttype == 'dct2d_8x8':
            # 8x8 block DCT-II / DCT-III
            self.transform_fn     = lambda x: block_transform(x, tdct.dct_2d, 8)
            self.inverse_fn       = lambda X: block_idtransform(X, tdct.idct_2d, 8)
        elif ttype == 'dct2d':
            # Full image DCT-II / DCT-III
            self.transform_fn     = tdct.dct_2d
            self.inverse_fn       = tdct.idct_2d
        else:
            raise ValueError(f"Unsupported transform type: {ttype}")

        registry = get_frequency_effect_registry()

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
    
    def inject(
        self,
        yuv: torch.Tensor,
        freq_jnd_map: torch.Tensor,
        channels: str = 'YUV',
        seed: Optional[int] = None,
        target: Optional[int] = 0,
        block_sz: Optional[int] = 8,
        sal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        In frequency domain, inject or reduce noise according to JND threshold, keep coeff sign unchanged, then IDCT back to pixel domain.

        target = 0 : Randomly add/subtract T_JND (original injection method)
                     Formula: C' = sign(C) * max(|C| + f * T_JND, 0), f ∈ {+1, -1} random
        target = 1 : Reduce amplitude (JPEG optimization mode), keep sign unchanged
                     Formula: C' = sign(C) * max(|C| - T_JND, 0)
        target = 2 : Weighted amplitude reduction
                     Formula:
                       C' = sign(C) * sqrt( max(C^2 - p(u,v)*JND^2, 0) )
                       And when |C| < JND, directly set to 0
        target = 3 : Consider salient regions, 
                     For non-salient regions, use amplitude reduction method from target == 2
                     For salient regions, use frequency domain energy enhancement filtering

        Args:
          yuv           : Original image, shape = (C, H, W)
          freq_jnd_map  : Frequency domain threshold, shape = (C, H, W)
          channels      : 'Y', 'UV', or 'YUV'
          seed          : Random seed (effective when target=0)
          target        : 0,1,2 three modes
          block_sz      : DCT block size (only used when target=2)

        Returns:
          out  : Image after injection/reduction, shape = (C, H, W)
        """
        C, H, W = yuv.shape
        out = yuv.clone()

        channels = self.channels
        seed     = self.seed
        target   = self.target
        block_sz = self.block_sz
        
        # 1) reproducible seed (only use random when target=0)
        if target == 0 and seed is not None:
            torch.manual_seed(seed)

        for c in range(C):
            # Determine if current channel needs processing
            do_ch = (channels == 'YUV') or \
                    (channels == 'Y'  and c == 0) or \
                    (channels == 'UV' and c in (1, 2))
            if not do_ch:
                continue

            # 2) Transform to frequency domain: coeff shape is (1, H, W)
            coeff = self.transform_fn(yuv[c:c+1])  # torch.Tensor

            # 3) Construct a block_sz×block_sz frequency weight p(u,v)
            p_block = torch.zeros((block_sz, block_sz),
                                   device=coeff.device,
                                   dtype=coeff.dtype)
            for i in range(block_sz):
                for j in range(block_sz):
                    s = i + j
                    if s == 0:
                        p_block[i, j] = 0.0
                    elif s <= 2:
                        p_block[i, j] = 0.25
                    elif 3 <= s <= 6:
                        p_block[i, j] = 0.75
                    else:
                        p_block[i, j] = 1.0

            # 4) Tile to full image size (H, W)
            nH = H // block_sz
            nW = W // block_sz
            # p_full: (H, W)
            p_full = p_block.repeat(nH, nW)
            # Convert to (1, H, W) for broadcasting
            p_map = p_full.unsqueeze(0)

            # 5) Take absolute value and original sign
            mag = coeff.abs()
            sign = coeff.sign()

            # 6) Choose different processing methods according to target
            thresh_c = freq_jnd_map[c].unsqueeze(0)  # (1, H, W)

            if target == 0:
                # Randomly add/subtract T_JND
                noise_sign = (torch.randint(
                    0, 2, coeff.shape, device=coeff.device, dtype=torch.int8
                ) * 2 - 1).to(coeff.dtype)  # Value is -1 or +1

                # new_mag = |C| + f * T_JND
                new_mag = mag + noise_sign * thresh_c
                new_mag = new_mag.clamp(min=0.0)

                coeff2 = sign * new_mag

            elif target == 1:
                # Reduce amplitude: C' = sign(C) * max(|C| - T_JND, 0)
                new_mag = mag - thresh_c
                new_mag = new_mag.clamp(min=0.0)

                coeff2 = sign * new_mag

            elif target == 2:
                mag2   = mag.pow(2)
                jnd2   = thresh_c.pow(2)
                inner  = (mag2 - p_map * jnd2).clamp(min=0.0)
                # Amplitude with sign
                coeff2 = torch.sqrt(inner) * sign
                # Set to 0 when |C| < JND
                coeff2 = coeff2 * (mag >= thresh_c)

            elif target == 3:
                # sal_mask might be numpy.ndarray, convert to PyTorch Tensor first
                if not isinstance(sal_mask, torch.Tensor):
                    sal_mask = torch.from_numpy(sal_mask).to(yuv.device)
                # Ensure it's bool type
                sal_mask = sal_mask.bool()

                # Downsample to get block-level saliency
                H_blk = H // block_sz
                W_blk = W // block_sz
                sal_blocks = (
                    sal_mask
                    .view(H_blk, block_sz, W_blk, block_sz)
                    .any(dim=(1,3))
                )  # (H_blk, W_blk) bool

                # —— SA blocks use equation (15), non-SA blocks use target=2 strategy —— #
                coeff2 = torch.empty_like(coeff)

                # First process coeff/gradients in blocks
                for by in range(H_blk):
                    for bx in range(W_blk):
                        y0, y1 = by*block_sz, (by+1)*block_sz
                        x0, x1 = bx*block_sz, (bx+1)*block_sz

                        block = coeff[:, y0:y1, x0:x1]    # (1,b,b)
                        m    = mag[:,   y0:y1, x0:x1]
                        sgn  = sign[:,  y0:y1, x0:x1]
                        t    = thresh_c[:,y0:y1, x0:x1]
                        pblk = p_block                 # (b,b)

                        if not sal_blocks[by, bx]:
                            # —— Non-salient, JND decay same as target=2 —— #
                            inner = (block.pow(2) - pblk * t.pow(2)).clamp(min=0.0)
                            b2    = torch.sqrt(inner) * sgn
                            b2    = b2 * (m >= t)
                            coeff2[:, y0:y1, x0:x1] = b2
                        else:
                            # —— Salient blocks: DCT coefficient enhancement —— #
                            #  (1) Calculate gradients using DCT coefficients
                            # horizontal gradient ∇^hor = 7.25·F(0,1) -2.55·F(0,3)
                            #                            +1.7·F(0,5) -1.44·F(0,7)
                            F = block.squeeze(0)
                            grad_h = abs(
                                  7.25 * F[0,1]
                                - 2.55 * F[0,3]
                                + 1.7  * F[0,5]
                                - 1.44 * F[0,7]
                            )
                            #  vertical gradient ∇^ver   similarly, use first column of odd rows
                            grad_v = abs(
                                  7.25 * F[1,0]
                                - 2.55 * F[3,0]
                                + 1.7  * F[5,0]
                                - 1.44 * F[7,0]
                            )
                            grad_sum = grad_h + grad_v + 1e-6

                            # (2) Calculate directional energy ratio sqrt(R^hor_v), sqrt(R^ver_u)
                            #     Here example uses all 1, can calculate according to paper (9–11)
                            # Original energy: horizontal/vertical frequency bands
                            #    Ω^hor_v is block[:, v], Ω^ver_u is block[u, :].
                            hor_energy = F.pow(2).sum(dim=0)   # shape (b,)
                            ver_energy = F.pow(2).sum(dim=1)   # shape (b,)

                            # 2) Prepare to store scaled energy and ratios
                            scaled_hor = torch.zeros_like(hor_energy)
                            scaled_ver = torch.zeros_like(ver_energy)
                            Rhor      = torch.ones_like(hor_energy)
                            Rver      = torch.ones_like(ver_energy)

                            # 3) Recursively calculate Rhor and scaled_hor
                            #    For n=0, define Rhor[0]=1, scaled_hor[0]=hor_energy[0]
                            scaled_hor[0] = hor_energy[0]
                            Rhor[0]      = 1.0
                            lambda_scale = 1.1
                            for v in range(1, block_sz):
                                # Sum of scaled energy of first v bands / sum of original energy
                                numer = scaled_hor[:v].sum()
                                denom = hor_energy[:v].sum()
                                Rhor[v] = numer / denom
                                if denom == 0:
                                    Rhor[v] = 1.0
                                else:
                                    Rhor[v] = scaled_hor[:v].sum() / denom
                                scaled_hor[v] = lambda_scale * Rhor[v] * hor_energy[v]


                            # 4) Similarly calculate Rver and scaled_ver
                            scaled_ver[0] = ver_energy[0]
                            Rver[0]      = 1.0

                            for u in range(1, block_sz):
                                numer = scaled_ver[:u].sum()
                                denom = ver_energy[:u].sum()
                                Rver[u] = numer / denom
                                if denom == 0:
                                    Rver[u] = 1.0
                                else:
                                    Rver[u] = scaled_ver[:v].sum() / denom
                                scaled_ver[u] = lambda_scale * Rver[u] * ver_energy[u]
    
                            # (3) Construct E_map (b, b)
                            # 1) Directional weight  
                            w_h = grad_h / grad_sum   # ∇^hor / ∇
                            w_v = grad_v / grad_sum   # ∇^ver / ∇

                            # 2) Square root of frequency band gain  
                            Rhor_sqrt = torch.sqrt(Rhor)    # (b,)
                            Rver_sqrt = torch.sqrt(Rver)    # (b,)
                            # 3) Expand them into (b,b) matrix  
                            #    First row is √Rhor, repeat along rows;
                            #    First column is √Rver, repeat along columns.
                            Rhor_mat = Rhor_sqrt.view(1, block_sz).expand(block_sz, block_sz)
                            Rver_mat = Rver_sqrt.view(block_sz, 1).expand(block_sz, block_sz)

                            # 4) Calculate E(u,v) according to equation (15)
                            #    E = √λ * ( w_h * √Rhor_mat + w_v * √Rver_mat )
                            E = (lambda_scale**0.5) * (w_h * Rhor_mat + w_v * Rver_mat)  # shape (b,b)
                            
                            E[0,0] = 1.0
                            E[1,0] = 1.0
                            E[1,1] = 1.0
                            E[0,1] = 1.0
                            E[0,2] = 1.0
                            E[2,0] = 1.0

                            # 5) Final enhancement: element-wise amplification of DCT coefficient block
                            block_enhanced = F * E  # block shape (b,b)

                            coeff2[:, y0:y1, x0:x1] = block_enhanced.unsqueeze(0)

            elif target == 4:
                # coeff: (1, H, W), thresh_c: (1, H, W)
                coeff2 = torch.empty_like(coeff)
            
                # Pre-construct ω_map and β
                b = block_sz
                # Here we use the simplest index approximation ω_{u,v} = sqrt(u^2+v^2)
                u_idx = torch.arange(b, device=coeff.device).view(b,1).expand(b,b).float()
                v_idx = torch.arange(b, device=coeff.device).view(1,b).expand(b,b).float()
                omega_map = torch.sqrt(u_idx**2 + v_idx**2)  # (b,b)
                beta = math.log(2**0.5) / (4 * math.pi**2)
            
                H_blk = H // b
                W_blk = W // b
            
                for by in range(H_blk):
                    for bx in range(W_blk):
                        y0, y1 = by*b, (by+1)*b
                        x0, x1 = bx*b, (bx+1)*b
            
                        blk    = coeff[:,    y0:y1, x0:x1].squeeze(0)  # (b,b)
                        J_blk  = thresh_c[:, y0:y1, x0:x1].squeeze(0)  # (b,b)
                        Cabs   = blk.abs() + 1e-6                        # Prevent division by zero
            
                        # 1) Calculate σ_d(u,v) frequency by frequency
                        ratio = J_blk / Cabs
                        # Limit ratio < 1, otherwise ln(1-ratio) will be invalid
                        ratio = torch.clamp(ratio, max=0.9999)
                        denom = -torch.log(1 - ratio)                   # (b,b)
                        sigma_uv = omega_map * torch.sqrt(beta / denom) # (b,b)
            
                        # 2) Block-level most stringent σ
                        sigma_blk = sigma_uv.max()
            
                        # 3) Generate gain G(u,v)
                        G = torch.exp(-beta * omega_map**2 / (sigma_blk**2))  # (b,b)
            
                        # 4) Multiply back coefficients
                        coeff2[:, y0:y1, x0:x1] = (blk * G).unsqueeze(0)

            else:
                raise ValueError("target must be 0 (random injection) \
                                              or 1 (magnitude reduction) \
                                              or 2 (weighted magnitude reduction) \
                                              or 3 (target 2 puls saliency enhancement)")

            # 5) Inverse transform back to pixel domain: rec shape (H, W)
            rec = self.inverse_fn(coeff2)[0]  # Get (H, W) from (1, H, W)
            out[c] = rec

        return out

    def forward(
        self,
        yuv: torch.Tensor
    ) -> torch.Tensor:
        """
        Default flow: compute → inject → return injected image
        Args:
          yuv      : Original input, shape = (C, H, W)
        Returns:
          contaminated : Image after injection, shape = (C, H, W)
        """
        freq_jnd_map = self.compute(yuv)  # (C, H, W)

        channels = self.channels
        seed     = self.seed
        target   = self.target
        block_sz = self.block_sz
        max_val  = 255.0

        contaminated = self.inject(yuv, freq_jnd_map, 
            channels=channels,
            seed=seed,
            target=target,
            block_sz=block_sz,
            sal_mask=get_block_saliency_mask(yuv[0]))

        jnd_map = torch.abs(yuv - contaminated)
        max_jnd = jnd_map.max()

        if max_jnd > 0:
            jnd_map = (jnd_map / max_jnd) * max_val

        return contaminated