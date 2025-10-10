#!/usr/bin/env python3
"""
Execute JND filtering on YUV or Y4M files according to specified start frame and frame count, and merge results into a single YUV/Y4M file

Usage examples:
  # Process .yuv
  python3 filt_video.py \
    --input ./BasketballPass.yuv --resolution 416x240 --format yuv420p --bitdepth 8 \
    --cfg inference/tmm_wu_2013.yml --start_frame 0 --frame_num 1 \
    --target 0 --platform cpu --cores 4 --output ./

  # Process .y4m
  python3 filt_video.py \
    --input ./video.y4m --cfg inference/frequency_jnd.yml --start_frame 10 --frame_num 3 \
    --target 0 --platform gpu --gpu_ids 0,1 --output ./
"""
import os
import re
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# local imports
from pyjnd.api_helpers import get_model
from pyjnd.utils import parse_y4m_header, yuvread2tensor, y4mread2tensor, tensor2yuv, tensor2y4m
from pyjnd.utils import imread2tensor, tensor2img, imwrite
from pyjnd.utils.options import parse_options

# Ensure multiprocessing spawn mode
multiprocessing.set_start_method('spawn', force=True)


def filter_frame(task):
    """Process single frame filtering task"""
    in_path, w, h, fmt, bd, idx, cfg_path, target, out_dir, platform, gpu_id = task
    # Device setup
    if platform == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Load configuration & model
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = get_model(cfg, device)

    # Read Tensor
    suffix = Path(in_path).suffix.lower()
    if suffix == '.yuv':
        yuv = yuvread2tensor(in_path, w, h, fmt=fmt, bitdepth=bd,
                             frame_idx=idx, normalize=cfg['normalized'])
    elif suffix == '.y4m':
        yuv = y4mread2tensor(in_path, frame_idx=idx, normalize=cfg['normalized'])
    else:
        yuv = imread2tensor(in_path, rgb=True)

    yuv = yuv.to(device)

    # pad to multiple of 32
    _, H0, W0 = yuv.shape
    pad_h = (-H0) % 32
    pad_w = (-W0) % 32
    if pad_h or pad_w:
        yuv = F.pad(yuv, (0, pad_w, 0, pad_h), mode='replicate')

    filt = model.forward(yuv)
    #!!! hack code: as learning based method would generate tensor with shape (B, C, H, W)    
    if filt.dim() == 4:
        filt = filt.squeeze(0)

    # Crop back to original size
    filt = filt[:, :H0, :W0]
    filt = filt.cpu()

    # Temporary output
    base = Path(in_path).stem
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"{base}_frame{idx}_filt{suffix}")
    if suffix == '.yuv':
        tensor2yuv(filt, tmp, fmt=fmt, bitdepth=bd, normalize=cfg['normalized'])
    elif suffix == '.y4m':
        tensor2y4m(filt, tmp, fmt=fmt, bitdepth=bd, normalize=cfg['normalized'])
    else:
        img = tensor2img(filt)
        imwrite(img, tmp)
    return tmp


def merge_yuv(paths, out_path):
    """Simple concatenation of multiple frame YUV"""
    with open(out_path, 'wb') as wf:
        for p in paths:
            with open(p, 'rb') as rf:
                wf.write(rf.read())
            os.remove(p)


def merge_y4m(paths, out_path):
    """Merge multiple frames according to Y4M specification, only keep first file header"""
    if not paths:
        return
    # Read first header
    with open(paths[0], 'rb') as f0:
        header = f0.readline()  # YUV4MPEG2 ...\n
    with open(out_path, 'wb') as wf:
        wf.write(header)
        # Write FRAME + data for each frame
        for p in paths:
            with open(p, 'rb') as rf:
                # Skip header
                rf.readline()
                # Write remaining FRAME\n + raw directly
                wf.write(rf.read())
            os.remove(p)


def sort_key(path):
    m = re.search(r'frame(\d+)', Path(path).stem)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Input file (.yuv or .y4m)")
    parser.add_argument("--resolution", help="WxH, can be omitted for .y4m")
    parser.add_argument("--format", help="Pixel format, yuv420p/... can be omitted for .y4m")
    parser.add_argument("--bitdepth", type=int, help="Bit depth, can be omitted for .y4m")
    parser.add_argument("--cfg", required=True, help="JND configuration file")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--frame_num", type=int, default=1)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--platform", choices=['cpu','gpu'], default='cpu')
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--gpu_ids", default="", help="GPU list, comma separated")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    in_path = args.input
    suffix = Path(in_path).suffix.lower()
    # Extract parameters
    if suffix == '.yuv':
        if not args.resolution or not args.format or args.bitdepth is None:
            parser.error('Must specify --resolution/--format/--bitdepth when processing .yuv')
        w, h = map(int, args.resolution.lower().split('x'))
        fmt, bd = args.format, args.bitdepth
    elif suffix == '.y4m':
        w, h, fmt, bd = parse_y4m_header(in_path)
    else:
        w, h, fmt, bd = (None, None, None, None)

    base = Path(in_path).stem
    out_dir = args.output

    # Build tasks
    tasks = []
    for idx in range(args.start_frame, args.start_frame + args.frame_num):
        tasks.append((in_path, w, h, fmt, bd, idx,
                      args.cfg, args.target, out_dir,
                      args.platform,
                      None if args.platform=='cpu' else
                      int(args.gpu_ids.split(',')[idx % len(args.gpu_ids.split(','))])
                     ))

    # Execute and collect
    temp_files = []
    if args.platform == 'cpu':
        with ProcessPoolExecutor(max_workers=args.cores) as exe:
            for out in exe.map(filter_frame, tasks):
                temp_files.append(out)
    else:
        # Simplified: GPU serial
        for t in tasks:
            temp_files.append(filter_frame(t))

    # Merge
    os.makedirs(out_dir, exist_ok=True)
    merged = os.path.join(out_dir, f"{base}_filt{suffix}")
    sorted_files = sorted(temp_files, key=sort_key)
    if suffix == '.yuv':
        merge_yuv(sorted_files, merged)
    elif suffix == '.y4m':
        merge_y4m(sorted_files, merged)
    else:
        if len(sorted_files) > 1:
            raise ValueError(f"{suffix} format images cannot be merged")

    print(f"[DONE] Filtering completed, output: {merged}")


if __name__ == "__main__":
    main()
