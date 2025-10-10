# Video Codec Preparation

This document describes how to install, configure, and run practical video encoders for testing our filtering algorithm. It covers three main sections:

1. Installing and compiling encoders.
2. Setting encoding parameters.
3. Executing the encoding workflow.

## 1. Installing and Compiling Encoders

We use a Makefile to manage encoder source retrieval and compilation:

### x264

```bash
make download_x264
make update_x264
```

### x265

```bash
make download_x265
make update_x265
```

### libaom (AV1)

```bash
make download_aom
make update_aom
```

### SVT-AV1

```bash
make download_svt_av1
make update_svt_av1
```

### VVenc (VVC)

```bash
make download_vvenc
make update_vvenc
```

### Decoders

* **AVC/HEVC**: decoded via FFmpeg (libx264, libx265)
* **AV1**: decoded via `aomdec` (from libaom)
* **VVC**: decoded via `vvdec`

```bash
make download_vvdec
make update_vvdec
```

## 2. Setting Encoding Parameters

Encoding parameters are managed via YAML configuration files located in the cfg/ directory. Each YAML file defines presets, rate-control settings, and other encoder-specific options for a particular encoder. When running the encoding workflow, specify the desired configuration file using the --config flag; the encoding script will parse the YAML file and apply all parameters automatically. The repository includes baseline YAML files for x264, x265, libaom, SVT-AV1, and VVenc, which contain common settings. You can customize or extend these files according to your testing requirements.
* **x264**: cfg/x264.yml
* **x265**: cfg/x265.yml
* **libaom**: cfg/aomenc.yml
* **SVT-AV1**: cfg/svt_av1.yml
* **vvenc**: cfg/vvenc.yml

In addition, you can refer to help/*.log to find help for encoder configuration

## 3. Executing Encoding Workflow

We use `encode.py` to automate encoding across a dataset based on user-defined YAML configs and CSV metadata. The script traverses each video entry in the specified CSV, applies the encoder configuration, and records quality scores for each decoded frame using the chosen metrics.

**Command Syntax:**

```bash
python3 encode.py \
  --encoder <config.yml> \
  --input_csv <dataset_meta.csv> \
  --input_dir <raw_yuv_folder> \
  --core <num_threads> \
  --output <result_folder> \
  --metrics <metric_1> <metric_2> ... \
  --db <sqlite_db_path> \
  --fps <frame_rate> \
  --frame_num <num_frames> \
  --enc_mode <CQP|CBR> \
  --qp_list <val1,val2,...>
```

**Example:**

```bash
python3 encode.py \
  --encoder ../codecs/cfg/x264.yml \
  --input_csv ../datasets/info/codec/hevc_sdr_ctc_meta_info.csv \
  --input_dir ../datasets/hevc_sdr_ctc/ori \
  --core 4 \
  --output ../codecs/result/hevc_sdr_ctc/ori/x264 \
  --metrics psnr psnr_hvsm ssim ms_ssim vmaf vmaf_neg \
  --db ../codecs/result/hevc_sdr_ctc/ori/x264/result.db \
  --fps 25 \
  --frame_num 1 \
  --enc_mode CQP \
  --qp_list 22,27,32,37
```

* `--encoder`: Path to the YAML config defining encoder parameters.
* `--input_csv`: CSV file listing raw YUV files and metadata.
* `--input_dir`: Directory containing raw input YUV sequences.
* `--core`: Number of CPU threads for parallel encoding.
* `--output`: Folder to save encoded bitstreams and decoded frames.
* `--metrics`: List of quality metrics to compute after decoding (e.g., `psnr`, `vmaf_neg`).
* `--db`: SQLite database file to store per-sequence metric results.
* `--fps`: Frame rate of the input video.
* `--frame_num`: Number of frames to encode and evaluate.
* `--enc_mode`: Encoding mode (`CQP` for constant QP, `CBR` for constant bitrate).
* `--qp_list`: Comma-separated list of QP values (or target bitrates when using `CBR`).

Alternatively, you can run `single_encode.sh`, a wrapper script for `encode.py`, with equivalent arguments.

For batch testing multiple filtering algorithms and encoders on the same dataset, use `regr_encode.sh`. In this script:

* Set `ENCODE_LIST=(x264 x265)` to iterate over multiple encoders (it will use `cfg/x264.yml`, `cfg/x265.yml`, etc.).
* Specify `DS_GROUP` for the dataset folder under `datasets/`. The script looks for subfolders matching each filter in `DATASET_LIST`.
* Use `ENC_MODE` to choose the encoding mode and `QP_LIST` (or target bitrate when `CBR`) to set coding parameters.
* `FRAME_NUM` controls number of frames per test, and `METRICS` defines the evaluation metrics.

Results are saved under:

```
result/$DS_GROUP/$DATASET/$ENCODER/
```

with bitstreams, decoded frames, and metric summaries in each encoder folder.


## 4. Calculating Compression Efficiency

After batch encoding completes, you can compare the compression efficiency between two encoding results using `calbdrate.py`, which leverages the Bjontegaard library to compute BD-BR (Bjøntegaard Delta Bit-Rate) and BD-PSNR (Bjøntegaard Delta PSNR).

**Example Command:**

```bash
python3 calbdrate.py \
  -a ../result/hevc_sdr_ctc/ori/x264/result.db \
  -t ../result/hevc_sdr_ctc/puc_he_2025/x264/result.db \
  -o puc_he_2025_x264.csv
```

* `-a <path>`: Path to the **anchor** SQLite result database (baseline encoder results).
* `-t <path>`: Path to the **test** SQLite result database (new encoder/filter results).
* `-o <file>`: Output CSV file for BD-BR and BD-PSNR comparison results.

For regression testing of multiple filters and encoders, use `regr_calbdrate.sh`. Its configurable parameters mirror those in `regr_encode.sh`:

* `ENCODE_LIST`: List of encoder names (e.g., `ENCODE_LIST=(x264 x265)`).
* `DS_GROUP`: Dataset group name under `datasets/`.
* `FILTER_LIST`: List of filter algorithm folders to compare.
* `ENC_MODE`: Encoding mode (`CQP` or `CBR`).
* `QP_LIST`: QP values or target bitrates (for CBR).
* `METRICS`: Evaluation metrics used for BD-rate computation.
* `DB_SUFFIX`: Suffix to identify anchor vs. test result databases.

The script will generate CSV summaries for each encoder and filter combination. 