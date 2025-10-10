#!/usr/bin/env python3
"""
Batch apply JND filtering to all YUV/Y4M files in a dataset according to specified frame count, and merge filtering results of each frame of the same video into a single YUV/Y4M file, output format consistent with input

Usage examples:
  python3 filt_one_dataset.py \
    --input_csv datasets/info/hevc_sdr_ctc_meta_info.csv \
    --input_dir datasets/hevc_sdr_ctc/ori \
    --cfg inference/tmm_wu_2013.yml \
    --frame_num 1 \
    --target 1 \
    --platform cpu --cores 4 \
    --output datasets/hevc_sdr_ctc/tmm_wu_2013

Or GPU:
  python3 filt_one_dataset.py \
    --input_csv datasets/info/hevc_sdr_ctc_meta_info.csv \
    --input_dir datasets/hevc_sdr_ctc/ori \
    --cfg inference/frequency_jnd.yml \
    --frame_num 3 \
    --target 0 \
    --platform gpu --gpu_ids 0,1 \
    --output ./filt_results
"""
import os
import re
import cv2
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Ensure multiprocessing spawn mode
multiprocessing.set_start_method('spawn', force=True)

# Import tools and models
from pyjnd.api_helpers import get_model
from pyjnd.utils import parse_y4m_header, yuvread2tensor, y4mread2tensor, tensor2yuv, tensor2y4m
from pyjnd.utils import imread2tensor, tensor2img, imwrite

def load_video_params(csv_path, input_dir):
    """
    Read video list from CSV and extract path and YUV/Y4M parameters for each file
    CSV must contain columns: filename, resolution (WxH), format (yuv420p/... or y4m), bitdepth
    - For .yuv files, parse resolution, format, bitdepth from CSV
    - For .y4m files, only record path here, auto-get by parse_y4m_header later
    Returns: [(fp, w, h, fmt, bd), ...], for .y4m, w/h/format/bitdepth can be None
    """
    df = pd.read_csv(csv_path)
    params = []
    for _, row in df.iterrows():
        fp = os.path.join(input_dir, row['filename'])
        if not os.path.isfile(fp):
            print(f"[WARN] File does not exist, skipping: {fp}")
            continue
        suffix = Path(row['filename']).suffix.lower()
        if suffix == '.yuv':
            # For YUV, CSV must contain resolution, format, bitdepth
            w, h = map(int, str(row['resolution']).split('x'))
            fmt = row['format']
            bd = int(row['bitdepth'])
        elif suffix == '.y4m':
            # For Y4M, placeholder None, auto-parse later
            w = h = None
            fmt = None
            bd = None
            continue
        else:
            # For JPG,PNG,BMP etc, placeholder None, auto-parse later
            w = h = None
            fmt = None
            bd = None
            continue
        params.append((fp, w, h, fmt, bd))
    return params


def sort_key(path):
    m = re.search(r'frame(\d+)', Path(path).stem)
    return int(m.group(1)) if m else -1


def filter_video(task):
    """
    Perform multi-frame JND filtering on a single video file (YUV/Y4M) and merge output
    task = (file_path, w, h, fmt, bd, frame_num, cfg_path, output_dir, platform, gpu_id)
    """
    file_path, w, h, fmt, bd, frame_num, cfg_path, output_dir, platform, gpu_id = task
    suffix = Path(file_path).suffix.lower()

    # For .y4m, override fmt, bd read from CSV; parse real header info
    if suffix == '.y4m':
        w, h, fmt, bd = parse_y4m_header(file_path)

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

    base = Path(file_path).stem
    os.makedirs(output_dir, exist_ok=True)
    temp_files = []

    # Calculate padding amount
    pad_h = (-h) % 32
    pad_w = (-w) % 32

    # Filter frame by frame
    for idx in range(frame_num):
        if suffix == '.yuv':
            yuv = yuvread2tensor(file_path, w, h, fmt=fmt, bitdepth=bd,
                                 frame_idx=idx, normalize=cfg['normalized'])
        elif suffix == '.y4m':
            yuv = y4mread2tensor(file_path, frame_idx=idx,
                                  normalize=cfg['normalized'])
        else:
            yuv = imread2tensor(file_path, rgb=True)
        yuv = yuv.to(device)

        # pad
        if pad_h or pad_w:
            yuv = F.pad(yuv, (0, pad_w, 0, pad_h), mode='replicate')

        # forward
        out = model.forward(yuv.unsqueeze(0) if yuv.dim()==3 else yuv)
        if out.dim()==4:
            out = out.squeeze(0)
        # crop
        out = out[:, :h, :w]
        out = out.cpu()

        # Temporary output
        ext = suffix
        tmp = os.path.join(output_dir, f"{base}_frame{idx}_filt{ext}")
        if suffix == '.yuv':
            tensor2yuv(out, tmp, fmt=fmt, bitdepth=bd,
                       normalize=cfg['normalized'])
        elif suffix == '.y4m':
            tensor2y4m(out, tmp, fmt=fmt, bitdepth=bd,
                       normalize=cfg['normalized'])
        else:
            img = tensor2img(out)
            imwrite(img, tmp)
        temp_files.append(tmp)
        print(f"[INFO] {base} frame{idx} processing completed")

    # Merge output
    merged = os.path.join(output_dir, f"{base}{suffix}")
    sorted_files = sorted(temp_files, key=sort_key)
    with open(merged, 'wb') as wf:
        if suffix == '.yuv':
            for p in sorted_files:
                wf.write(open(p,'rb').read())
                os.remove(p)
        elif suffix == '.y4m':
            # Y4M: Keep first frame header
            with open(sorted_files[0],'rb') as f0:
                wf.write(f0.readline())
            for p in sorted_files:
                with open(p,'rb') as fp:
                    fp.readline()
                    wf.write(fp.read())
                os.remove(p)
        else:
            # Image format does not support merging
            if len(sorted_files) > 1:
                raise ValueError(f"{suffix} format images cannot be merged") 

    print(f"[DONE] {base} all frames merged to {merged}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch execute JND filtering on all YUV/Y4M files in dataset"
    )
    parser.add_argument("--input_csv", required=True,
                        help="CSV metadata, contains filename, resolution, format, bitdepth columns")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--frame_num", type=int, default=1)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--platform", choices=["cpu","gpu"], default="cpu")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default="",
                        help="GPU parallel, comma-separated device index list")
    parser.add_argument("--output", type=str, default="./filt_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = load_video_params(args.input_csv, args.input_dir)
    if not videos:
        print("[ERROR] No video files loaded")
        return

    tasks = []
    if args.platform == 'cpu':
        for fp, w, h, fmt, bd in videos:
            tasks.append((fp, w, h, fmt, bd,
                          args.frame_num, args.cfg, args.output,
                          'cpu', None))
        with ProcessPoolExecutor(max_workers=args.cores) as exe:
            exe.map(filter_video, tasks)
    else:
        gpus = [int(x) for x in args.gpu_ids.split(',')]
        for i, (fp, w, h, fmt, bd) in enumerate(videos):
            gid = gpus[i % len(gpus)]
            tasks.append((fp, w, h, fmt, bd,
                          args.frame_num, args.cfg, args.output,
                          'gpu', gid))
        with ProcessPoolExecutor(max_workers=len(gpus)) as exe:
            exe.map(filter_video, tasks)

    print("[ALL DONE] All videos filtered and merged!")

if __name__ == "__main__":
    main()