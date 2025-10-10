import subprocess
import argparse
import os
import csv
import re
import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pickle as pkl
import decord
import yaml
import shutil
import filecmp
import math

from tqdm import tqdm
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--encoder", type=str, default="./cfgBhv/svt_av1.yml", help="the configure file of encoder")
    parser.add_argument("--enc_mode"     , type=str, default="CQP", help="the mode of encoder, CQP, CBR and CRF modes are supported")
    parser.add_argument("--qp_list"      , type=str, default="22,27,32,37", help="comma-separated list of QP/CR/RF values")
    parser.add_argument("--frame_num"    , type=int , default=1, help="the number of frame to be encoded")
    parser.add_argument("--fps"          , type=int , default=25, help="frame per second")
    parser.add_argument("--input_csv"    , type=str, default="../../dataset/info/svac_meta_info.csv", help="the meta infomation of source video")
    parser.add_argument("--input_dir"    , type=str, default="../../dataset/svac/", help='the directory where YUV videos are located')
    parser.add_argument("-c", "--core"   , type=int, default=7, help='the core number used to parallelly encode source videos')
    parser.add_argument("-m", "--metrics", nargs="+", choices=["psnr", "psnr_hvsm", "ssim", "ms_ssim", "ssimulacra2", "vmaf", "vmaf_neg"], default=["psnr"], help="List of metrics to calculate")
    parser.add_argument("-o", "--output" , type=str, default="../../dataset/compressed" , help='the output directory to store compressed video')
    parser.add_argument("--db", type=str , default="results.db", help="SQLite3 database storage file")
    args = parser.parse_args()
    return args

def check_or_create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"[INFO] Output directory created: {output_path}")
    else:
        print("[INFO] Proceeding and overwriting the existing directory.")

def csv_load(csvPath, input_dir):
    filehandle = pd.read_csv(csvPath)
    files = filehandle['filename']
    resolutions = filehandle['resolution']
    bitDepths = filehandle['bitdepth']
    formats = filehandle['format']
    numbers = filehandle['number']
    
    # Use input_dir to build complete path
    full_path_files = [os.path.join(input_dir, str(file)) for file in files]
    
    widths = []
    heights = []
    for resolution in resolutions:
        width, height = resolution.split('x')
        widths.append(width)
        heights.append(height)
    
    videoParam = {}
    for i in range(len(full_path_files)):
        file = full_path_files[i]
        videoParam[file] = { 'filename': files[i], 'width': widths[i], 'height': heights[i], 'bitDepth': bitDepths[i], 'format': formats[i], 'number': numbers[i] }
    return videoParam

def get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio):
    # get display time
    time = float(encNumber) / float(encFps)
    
    # get size in bit
    size = int(width) * int(height) * int(encNumber) * 8 * 1.5 / float(ratio)
    
    # get rate in bps
    rate = float(size) / float(time) / 1000.0
    return int(rate)

def set_enc_param(encParam, videoType, videoPath, videoName, width, height, bitDepth, format_, encFps, encNumber, ratio):
    if encParam['name'] == 'jpeg':
        encParam['video_param']['--input'] = videoPath
        encParam['video_param']['--input-res'] = f"{width}x{height}"
        encParam['encoder_param']['--output'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        output_bin = encParam['encoder_param']['--output']

        if args.enc_mode == "CQP":
            encParam['encoder_param']['--quality'] = ratio
        elif args.enc_mode == "CBR":
            print(f"[ERROR] {encParam['name']} doesn't support {args.enc_mode}")
        elif args.enc_mode == "CRF":
            print(f"[ERROR] {encParam['name']} doesn't support {args.enc_mode}")
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")

    elif encParam['name'] == 'x264':
        encParam['video_param']['--input'] = videoPath
        encParam['video_param']['--input-res'] = f"{width}x{height}"
        encParam['encoder_param']['--frames'] = encNumber
        encParam['encoder_param']['--dump-yuv'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}_enc.yuv"
        encParam['encoder_param']['-o'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        output_bin = encParam['encoder_param']['-o']

        if args.enc_mode == "CQP":
            encParam['encoder_param'].pop('--bitrate', None)
            encParam['encoder_param'].pop('--crf', None)
            encParam['encoder_param']['--qp'] = ratio
        elif args.enc_mode == "CBR":
            encParam['encoder_param'].pop('--qp', None)
            encParam['encoder_param'].pop('--crf', None)
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--bitrate'] = bitrate
        elif args.enc_mode == "CRF":
            encParam['encoder_param'].pop('--bitrate', None)
            encParam['encoder_param'].pop('--qp', None)
            encParam['encoder_param']['--crf'] = ratio
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")
            
    elif encParam['name'] == 'x265':
        encParam['video_param']['--input'] = videoPath
        encParam['video_param']['--input-res'] = f"{width}x{height}"
        encParam['encoder_param']['--frames'] = encNumber
        encParam['encoder_param']['--output'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        encParam['encoder_param']['--recon'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}_enc.yuv"
        encParam['encoder_param']['--csv'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.csv"
        output_bin = encParam['encoder_param']['--output']

        if args.enc_mode == "CQP":
            encParam['encoder_param'].pop('--bitrate', None)
            encParam['encoder_param'].pop('--crf', None)
            encParam['encoder_param']['--qp'] = ratio
        elif args.enc_mode == "CBR":
            encParam['encoder_param'].pop('--qp', None)
            encParam['encoder_param'].pop('--crf', None)
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--bitrate'] = bitrate
        elif args.enc_mode == "CRF":
            encParam['encoder_param'].pop('--bitrate', None)
            encParam['encoder_param'].pop('--qp', None)
            encParam['encoder_param']['--crf'] = ratio
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")

    elif encParam['name'] == 'vvenc':
        encParam['video_param']['--InputFile'] = videoPath
        encParam['video_param']['--SourceWidth'] = width
        encParam['video_param']['--SourceHeight'] = height
        encParam['encoder_param']['--FramesToBeEncoded'] = encNumber
        encParam['encoder_param']['--InternalBitDepth'] = bitDepth
        encParam['encoder_param']['--OutputBitDepth'] = bitDepth
        encParam['encoder_param']['--ReconFile'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}_enc.yuv"
        encParam['encoder_param']['--BitstreamFile'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        output_bin = encParam['encoder_param']['--BitstreamFile']

        if args.enc_mode == "CQP":
            encParam['encoder_param'].pop('--TargetBitrate', None)
            encParam['encoder_param']['--QP'] = ratio
        elif args.enc_mode == "CBR":
            encParam['encoder_param'].pop('--QP', None)
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--TargetBitrate'] = bitrate * 1000
        elif args.enc_mode == "CRF":
            print(f"[ERROR] {encParam['name']} doesn't support {args.enc_mode}")
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")

    elif encParam['name'] == 'svt_av1':
        encParam['video_param']['--input'] = videoPath
        encParam['video_param']['--width'] = width
        encParam['video_param']['--height'] = height
        encParam['encoder_param']['--frames'] = encNumber
        encParam['encoder_param']['--recon'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}_enc.yuv"
        encParam['encoder_param']['--output'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        output_bin = encParam['encoder_param']['--output']

        if args.enc_mode == "CQP":
            encParam['encoder_param'].pop('--tbr', None)
            encParam['encoder_param']['--rc'] = 0
            encParam['encoder_param']['--aq-mode'] = 0
            encParam['encoder_param']['--qp'] = ratio
        elif args.enc_mode == "CBR":
            encParam['encoder_param']['--rc'] = 2
            encParam['encoder_param']['--aq-mode'] = 2
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--tbr'] = bitrate
        elif args.enc_mode == "CRF":
            encParam['encoder_param'].pop('--tbr', None)
            encParam['encoder_param']['--rc'] = 0
            encParam['encoder_param']['--aq-mode'] = 2
            encParam['encoder_param']['--qp'] = ratio
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")
    
    elif encParam['name'] == 'aomenc':
        encParam['video_param']['--input'] = videoPath
        encParam['video_param']['--width'] = width
        encParam['video_param']['--height'] = height
        encParam['encoder_param']['--limit'] = encNumber
        # encParam['encoder_param']['--recon'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}_enc.yuv"
        encParam['encoder_param']['--output'] = args.output + f"/{videoType}_{videoName}_{ratio}_{encParam['name']}.bin"
        output_bin = encParam['encoder_param']['--output']

        if args.enc_mode == "CQP":
            encParam['encoder_param'].pop('--target-bitrate', None)
            encParam['encoder_param']['--end-usage'] = "q"
            encParam['encoder_param']['--cq-level'] = ratio
        elif args.enc_mode == "CBR":
            encParam['encoder_param']['--end-usage'] = "cbr"
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--target-bitrate'] = bitrate
        elif args.enc_mode == "CRF":
            bitrate = get_bit_rate(width, height, bitDepth, format_, encFps, encNumber, ratio)
            encParam['encoder_param']['--target-bitrate'] = bitrate
            encParam['encoder_param']['--end-usage'] = "cq"
            encParam['encoder_param']['--cq-level'] = ratio
        else:
            print(f"[ERROR] Unsupported encode mode {args.enc_mode}")

    encParam['output_path'] = output_bin
    return encParam


def cmd_wrapper(command, encParam):
    if encParam['name'] == 'jpeg':
        command.extend([str(encParam['video_param']['--input'])])
        command.extend([str(encParam['video_param']['--input-res'])])
        command.extend([str(encParam['encoder_param']['--output'])])
        command.extend([str(encParam['encoder_param']['--quality'])])
        command.extend([str(encParam['encoder_param']['--optimize'])])

    elif encParam['name'] == 'x264':
        # add encoder parameters
        for param, value in encParam['encoder_param'].items():
            command.extend([param, str(value)])

        # add video parameters
        for param, value in encParam['video_param'].items():
            if param != '--input':
                command.extend([param, str(value)])
        
        # add video input
        command.extend([str(encParam['video_param']['--input'])])
    elif encParam['name'] == 'x265' or encParam['name'] == 'svt_av1':
        # add encoder parameters
        for param, value in encParam['encoder_param'].items():
            command.extend([param, str(value)])

        # add video parameters
        for param, value in encParam['video_param'].items():
            command.extend([param, str(value)])
    elif encParam['name'] == 'aomenc':
        # add output and input path
        command.extend(["--output" + "=" + str(encParam['encoder_param']['--output'])])
        command.extend([str(encParam['video_param']['--input'])])

        # add encoder parameters
        for param, value in encParam['encoder_param'].items():
            if param != '--output':
                command.extend([param + "=" + str(value)])
        command.extend(["--rt"])
        command.extend(["--psnr"])
        command.extend(["--ivf"])
        command.extend(["--disable-warning-prompt"])
        command.extend(["-y"])
            
        # add video parameters
        for param, value in encParam['video_param'].items():
            if param != '--input':
                command.extend([param + "=" + str(value)])

    elif encParam['name'] == 'vvenc':
        # add encoder parameters
        for param, value in encParam['encoder_param'].items():
            if param == "-c":
                command.extend([param, str(value)])
            else:
                command.extend([param + "=" + str(value)])

        # add video parameters
        for param, value in encParam['video_param'].items():
            command.extend([param + "=" + str(value)])
    return command

def compute_bpp(bitstream_path, num_frames, width, height):
    size_bytes = os.path.getsize(bitstream_path)
    bpp = (int(size_bytes) * 8) / (int(num_frames) * int(width) * int(height))
    return bpp

def get_dataset_name(path):
    # Split by system separator
    parts = path.split(os.sep)
    if "datasets" in parts:
        idx = parts.index("datasets")
        # Ensure there's another segment after
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

# —— Call ffmpeg to calculate metrics —— #
def calc_metric(metric, orig_yuv, dec_yuv, width, height, fps, encFormat, encNum):

    target_prefix = get_dataset_name(orig_yuv) + '/'
    new_dir = "ori"
    s = orig_yuv
    index = s.find(target_prefix)
    if index == -1:
        new_s = s
    else:
        start = index + len(target_prefix)
        end = s.find('/', start)
        new_s = s[:start] + new_dir + s[end:]
    orig_yuv = new_s

    if metric == "psnr":
        cmd = [
            "ffmpeg", "-y",
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", orig_yuv, 
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", dec_yuv, 
            "-frames:v", encNum,
            "-lavfi", "psnr",
            "-f", "null", "-"
        ]
        pattern = r"average:([0-9]+\.[0-9]+)"

    elif metric == "ssim":
        cmd = [
            "ffmpeg", "-y",
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", orig_yuv, 
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", dec_yuv, 
            "-frames:v", encNum,
            "-lavfi", "ssim",
            "-f", "null", "-"
        ]
        pattern = r"All:([0-9]+\.[0-9]+)"

    elif metric == "vmaf":
        cmd = [
            "ffmpeg", "-y",
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", dec_yuv, 
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", orig_yuv, 
            "-frames:v", encNum,
            "-lavfi", "libvmaf=model='path=/usr/share/model/vmaf_v0.6.1.json':log_fmt=json",
            "-f", "null", "-"
        ]
        pattern = r"VMAF score\s*:\s*([0-9]+\.[0-9]+)"

    elif metric == "vmaf_neg":
        cmd = [
            "ffmpeg", "-y",
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", dec_yuv, 
            "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat, "-i", orig_yuv, 
            "-frames:v", encNum,
            "-lavfi", "libvmaf=model='path=/usr/share/model/vmaf_v0.6.1neg.json':log_fmt=json",
            "-f", "null", "-"
        ]
        pattern = r"VMAF score\s*:\s*([0-9]+\.[0-9]+)"

    elif metric == "ssimulacra2":
        # 1) Generate temporary JPEG path
        orig_jpg  = os.path.splitext(dec_yuv)[0]  + "_orig.jpg"
        dec_jpg = os.path.splitext(dec_yuv)[0] + "_dec.jpg"

        # 2) Use jpeg to convert YUV → JPEG
        subprocess.run(
            ["../bin/jpeg", orig_yuv, f"{width}x{height}", orig_jpg, "100", "1"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["../bin/jpeg", dec_yuv, f"{width}x{height}", dec_jpg, "100", "1"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 3) Call ssimulacra2, read stdout
        proc = subprocess.run(
            ["../bin/ssimulacra2", orig_jpg, dec_jpg],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        # 4) Output should be a pure number, e.g. "81.53383798\n"
        score = float(proc.stdout.strip())

        # Delete temporary files
        os.remove(orig_jpg)
        os.remove(dec_jpg)
        return score

    elif metric == "psnr_hvsm":
        csv_file = os.path.splitext(dec_yuv)[0]
        # Call VQMT to calculate PSNR-HVSM and generate CSV
        cmd = [
            "../bin/vqmt", orig_yuv, dec_yuv,
            str(height), str(width), str(encNum), "1",
            csv_file, "PSNRHVSM"
        ]
        subprocess.run(cmd, check=True)
        # Parse average
        score = None
        with open(csv_file + '_psnrhvsm.csv') as f:
            for line in f:
                if line.startswith('average'):
                    score = float(line.strip().split(',')[1])
                    break
        os.remove(csv_file + '_psnrhvsm.csv')
        return score

    elif metric == "ms_ssim":
        csv_file = os.path.splitext(dec_yuv)[0]

        # 1) Generate temporary yuv path, after padding
        orig_yuv_crop  = os.path.splitext(dec_yuv)[0]  + "_orig.yuv"
        dec_yuv_crop = os.path.splitext(dec_yuv)[0] + "_dec.yuv"
        crop_w = (int(width) // 16) * 16
        crop_h = (int(height) // 16) * 16

        # 2) Use ffmpeg to pad yuv width and height to be divisible by 16
        if crop_w != int(width) or crop_h != int(height):
            subprocess.run(
                ["ffmpeg", "-y", "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat,
                 "-i", orig_yuv, "-vf", f"crop={crop_w}:{crop_h}:x=0:y=0", "-frames:v", str(encNum), 
                 "-c:v", "rawvideo", "-pix_fmt", encFormat, orig_yuv_crop],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            subprocess.run(
                ["ffmpeg", "-y", "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", encFormat,
                 "-i", dec_yuv, "-vf", f"crop={crop_w}:{crop_h}:x=0:y=0", "-frames:v", str(encNum), 
                 "-c:v", "rawvideo", "-pix_fmt", encFormat, dec_yuv_crop],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            orig_yuv_crop = orig_yuv
            dec_yuv_crop = dec_yuv

        # Call VQMT to calculate MSSSIM and generate CSV
        cmd = [
            "../bin/vqmt", orig_yuv_crop, dec_yuv_crop,
            str(crop_w), str(crop_h), str(encNum), "1",
            csv_file, "MSSSIM"
        ]
        score = None
        try:
            subprocess.run(cmd, check=True)
            with open(csv_file + '_msssim.csv') as f:
                for line in f:
                    if line.startswith('average'):
                        score = float(line.strip().split(',')[1])
                        break
        except subprocess.CalledProcessError:
            print(f"[ERROR] ms_ssim calculation failed for {dec_yuv}")

        # Delete temporary files
        if crop_w != int(width) or crop_h != int(height):
            os.remove(orig_yuv_crop)
            os.remove(dec_yuv_crop)
        os.remove(csv_file + '_msssim.csv')

        return score
        
    else:
        return None

    # Execute and parse stderr
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    stderr = proc.stderr
    match = re.search(pattern, stderr)

    return float(match.group(1))

def decode_video(encoded_path, encoder_type, pix_fmt, resolution):
    """
    Decode encoder output bitstream file to YUV.
    - encoded_path: e.g. "../../video1_10_x265.bin"
    - encoder_type: "x264" / "x265" / "svt_av1" / "vvenc"
    - pix_fmt: value read from CSV format column, e.g. "yuv420p", "yuv420p10le", etc.
    - resolution: value assembled from CSV, e.g. "1920x1080"
    """
    decoded_path = os.path.splitext(encoded_path)[0] + "_dec.yuv"

    if encoder_type in ["jpeg", "x264", "x265"]:
        cmd = [
            "ffmpeg",
            "-y",                   # Force overwrite
            "-i", encoded_path,     # Input
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,    # Dynamically specify
            "-s", resolution,       # Dynamically specify
            decoded_path
        ]
    elif encoder_type in ["svt_av1", "aomenc"]:
        cmd = [
            "../bin/aomdec",
            "-v", encoded_path,
            "--i420",
            "-o", decoded_path
        ]
    elif encoder_type == "vvenc":
        cmd = [
            "../bin/vvdec",
            "-b", encoded_path,
            "-o", decoded_path
        ]
    else:
        print(f"[WARN] Unsupported encoder '{encoder_type}', skip decode.")
        return None

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return decoded_path
    except subprocess.CalledProcessError:
        print(f"[ERROR] Decoding failed for {encoded_path}")
        return None

def encode_video(key, videoParam, encParam, videoType, args, missing_files):
    if not os.path.exists(key):
        print(f"[Warning] YUV file does not exist: {key}")
        missing_files.append(key)
        return

    video = os.path.splitext(os.path.basename(key))[0]
    width = videoParam[key]['width']
    height = videoParam[key]['height']
    bitDepth = videoParam[key]['bitDepth']
    format_ = videoParam[key]['format']
    allNumber = videoParam[key]['number']
    encFps = args.fps
    encNumber = args.frame_num
    assert allNumber >= encNumber, f"the number of frame of {key} is less than {encNumber}"
    assert int(bitDepth) == 8, f"the bit depth of {key} is {bitDepth}, which is not supported now"
    assert format_ == "yuv420p",  f"the yuv format of {key} is {format_}, which is not supported now"
    args.qp_list = [int(x) for x in args.qp_list.split(',')]
    pbar = tqdm(args.qp_list)
    pbar.set_description("Encoding %s" % video)
    for ratio in pbar:
        # set encParam
        encParam = set_enc_param(encParam, videoType, key, video, width, height, bitDepth, format_, encFps, encNumber, ratio)

        # build command line
        command = ["../bin/" + encParam['name']]
        command = cmd_wrapper(command, encParam)

        # run encoder
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        pix_fmt    = videoParam[key]['format']               # CSV format column
        resolution = f"{width}x{height}"                     # CSV resolution column
        try:
            decoded = decode_video(encParam['output_path'], encParam['name'], pix_fmt, resolution)
        except Exception as e:
            print(f"[ERROR] Decoding failed for {encParam['output_path']}: {e}")

        # Construct "encoded yuv" path: same prefix, suffix changed from _dec.yuv to _enc.yuv
        enc_yuv = decoded.replace("_dec.yuv", "_enc.yuv")
        if  os.path.exists(enc_yuv):
            # Actually do byte-by-byte comparison of file content
            if not filecmp.cmp(enc_yuv, decoded, shallow=False):
                print(f"[ERROR] YUV mismatch → {enc_yuv} and {decoded} content inconsistent!")
            else:
                pass

        # Calculate Bpp
        bs_file = encParam['output_path']
        bpp = compute_bpp(bs_file, encNumber, width, height)
        logging.info(f"{bs_file} BPP={bpp:.6f}")

        # Iterate through metrics for calculation
        result = {"video": videoParam[key]['filename'], "enc_ratio": ratio, "bpp": bpp}
        for m in args.metrics:
            val = calc_metric(m, key, decoded, width, height, encFps, videoParam[key]['format'], str(encNumber))
            result[m.replace("-", "_")] = val
            logging.info(f"{bs_file} {m}={val}")

        # Insert into SQLite
        c.execute("""
            INSERT INTO results (video, enc_ratio, bpp, psnr, psnr_hvsm, ssim, ms_ssim, ssimulacra2, vmaf, vmaf_neg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["video"], result["enc_ratio"], result["bpp"],
            result.get("psnr"), result.get("psnr_hvsm"), result.get("ssim"), result.get("ms_ssim"), result.get("ssimulacra2"), result.get("vmaf"), result.get("vmaf_neg")
        ))
        conn.commit()
        
        if  os.path.exists(enc_yuv):
            os.remove(enc_yuv)
        os.remove(decoded)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[
                            logging.FileHandler("run.log", mode="w"),
                            logging.StreamHandler()
                        ])

    check_or_create_output_dir(args.output)

    # Create SQLite database
    if os.path.exists(args.db):
        os.remove(args.db)
    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE results (
            video TEXT,
            enc_ratio INTEGER,
            bpp REAL,
            psnr REAL,
            psnr_hvsm REAL,
            ssim REAL,
            ms_ssim REAL,
            ssimulacra2 REAL,
            vmaf REAL,
            vmaf_neg REAL
        )
    """)
    conn.commit()

    with open(args.encoder, "r") as f:
        encParam = yaml.safe_load(f)

    videoParam = csv_load(args.input_csv, args.input_dir)
    videoType = args.input_dir.split('/')[-1]

    manager = Manager()
    missing_files = manager.list()

    with ProcessPoolExecutor(max_workers=args.core) as executor:
        futures = [executor.submit(encode_video, key, videoParam, encParam, videoType, args, missing_files)
                   for key in videoParam.keys()]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()

    if missing_files:
        print("\n[Summary] The following YUV files are missing:")
        for path in missing_files:
            print(f"  - {path}")

    conn.close()
