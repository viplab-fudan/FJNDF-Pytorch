# model_stats.py
# Script to count model parameters (MB) and computational complexity (GFLOPS)
# Usage example: python model_stats.py --cfg-path ./config.yml --input-shape 1 3 256 256

import argparse
import yaml
import torch
from ptflops import get_model_complexity_info
from pyjnd.api_helpers import get_model

def main():
    parser = argparse.ArgumentParser(description="Script to count model parameters and GFLOPS")
    parser.add_argument('--cfg-path', type=str, required=True, help='Model configuration YAML file path, including architecture and weight information')
    parser.add_argument('--input-shape', type=int, nargs=4, required=True,
                        metavar=('N','C','H','W'), help='Example input shape')
    parser.add_argument('--device', type=str, default='cpu', help='Running device, e.g. cpu or cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    # Read configuration and instantiate model (including loading weights)
    with open(args.cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = get_model(cfg, device=device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count FLOPs
    _, C, H, W = args.input_shape
    flops, _ = get_model_complexity_info(
        model,
        (C, H, W),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    gflops = flops * 2 / 1e9

    # Output results
    print(f"Configuration file: {args.cfg_path}")
    print(f"Parameters: {total_params:,} K")
    print(f"Computational complexity: {gflops:.2f} GFLOPS")

if __name__ == '__main__':
    main()