# 🔍 Frequency-Domain JND-Guided Pre-Filter Benchmark

This project provides a **Frequncy-Domain JND-Guided (Just Noticeable Difference) Pre-Filter Benchmark** to evaluate the performance of various combinations of frequency-domain JND modeling and injection algorithms in the pre-processing stage of video compression. The platform supports batch filtering of YUV/Y4M video sequences and generates standardized experiment results.

---

## 🛠️ Environment Setup

This project is developed with **Python 3.8**. We recommend using a virtual environment (e.g., `virtualenv` or `conda`) to avoid dependency conflicts.

```bash
conda create -n myenv python=3.8 -y
conda activate myenv
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

This project depends on PyTorch 2.4.1 with CUDA 12.4. Make sure your hardware supports this version. If you are using CPU only or a different CUDA version, please refer to the PyTorch official website for the appropriate installation command.

## 📂 Data Preparation

### 1. Input File Organization

This platform currently supports reading, processing, and outputting YUV/Y4M files. Place your YUV/Y4M files in the `datasets/` directory. For YUV files, provide metadata (resolution, bit depth, format, number of frames) in a CSV file located under `datasets/codec/info/`. For Y4M files, only a filename column is required.

### 2. CSV Format Requirements

The CSV file must include the following headers:

```
filename,resolution,bitdepth,format,number
```

Each row represents one YUV file, for example:

```
Kimono.yuv,1920x1080,8,yuv420p,240
```

Refer to `datasets/info/codec/hevc_sdr_ctc_meta_info.csv` for an example. If your files are in Y4M format, the CSV only needs a `filename` column.

### 3. Dataset Download and Preparation

For detailed instructions on downloading and preparing datasets, please refer to [`datasets/prepare_datasets.md`](datasets/prepare_datasets.md).

## 🚀 Usage

This package provides two scripts for JND filtering, one for individual videos and one for batch processing.

### 1. Filter a Single YUV Video

Use `filt_one_video.py` to filter a single YUV file:

```bash
python3 filt_one_video.py \
    --input ./assets/kodim01.yuv \
    --resolution 768x512 \
    --format yuv420p \
    --bitdepth 8 \
    --cfg options/inference/infer_iccv_yan_lite_2025.yml \
    --start_frame 0 \
    --frame_num 1 \
    --platform cpu \
    --cores 1 \
    --gpu_ids 0 \
    --output ./
```

| Parameter      | Description                                                                                |
| -------------- | ------------------------------------------------------------------------------------------ |
| `--input`      | Path to the input YUV/Y4M file                                                             |
| `--resolution` | Video resolution in the format `WIDTHxHEIGHT` (e.g., `416x240`)                            |
| `--format`     | YUV sampling format. Supported: `yuv400p`, `yuv420p`, `yuv422p`, `yuv444p`                 |
| `--bitdepth`   | Bit depth per pixel, e.g., `8`, `10`                                                       |
| `--cfg`        | Path to the JND filter configuration file (YAML), located in `options/inference/`          |
| `--start_frame`| Starting frame index (0-based)                                                             |
| `--frame_num`  | Number of consecutive frames to process from the start frame                               |
| `--platform`   | Execution platform. Currently only `cpu` is supported                                      |
| `--cores`      | Maximum number of CPU cores for parallel processing                                        |
| `--gpu_ids`    | GPU device IDs for parallel processing when platform is `gpu`                              |
| `--output`     | Output directory where the filtered YUV/Y4M files will be stored                           |

You can also refer to `run.sh`, which wraps the usage of `filt_one_video.py`.

### 2. Batch Filter a Folder of YUV Videos

Use `filt_one_dataset.py` to process all YUV files in a folder:

```bash
python3 filt_one_dataset.py \
    --input_csv datasets/info/codec/hevc_sdr_ctc_meta_info.csv \
    --input_dir datasets/hevc_sdr_ctc/ori \
    --cfg options/inference/infer_iccv_yan_lite_2025.yml \
    --frame_num 1 \
    --platform cpu \
    --cores 4 \
    --gpu_ids 0,1 \
    --output datasets/hevc_sdr_ctc/iccv_yan_lite_2025/
```

| Parameter        | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `--input_csv`    | Path to the CSV file containing metadata for each YUV/Y4M file         |
| `--input_dir`    | Directory containing the input YUV video files                         |
| `--cfg`          | Path to the JND filter configuration file (YAML), located in `options/inference/` |
| `--frame_num`    | Number of frames to process per video (starting from frame 0)         |
| `--platform`     | Execution platform. Supported: `cpu`, `gpu`                            |
| `--cores`        | Maximum number of CPU cores for parallel processing (effective for `cpu`) |
| `--gpu_ids`      | GPU device IDs for parallel processing (effective for `gpu`)          |
| `--output`       | Output directory for all filtered video files                         |

Refer to `regr_filt.sh`, which provides a wrapper for `filt_one_dataset.py`.


## 📂 Codec Preparation

To assess the practical impact of our filtering algorithm, we have integrated and tested it with the following widely-used video encoders:

- **x264**  
- **x265**  
- **libaom**
- **VVenc**

For detailed instructions on encoder installation, configuration, and usage, please refer to [`codecs/prepare_video_codecs.md`](codecs/prepare_video_codecs.md).  

## 🔬 Validation Results

We have validated our algorithm performance on the JND Filtering Benchmark Platform. Detailed test results are available in the [`results/results.md`](results/results.md).

## 📂 Train Pipeline

### 1. Prepare Image or Video Dataset for Training
You need to prepare the original YUV data for training. If you wish to use a supervised learning approach, you will also need to provide reference YUV data. For reference, we provide the `divkon_2k` dataset, which combines two popular datasets: `div2k` and `konjnd-1k`. In addition to the original YUV data, we also provide reference YUV data that has undergone multiple degradation and detail enhancement processes, suitable for supervised learning. For details, please refer to [`datasets/prepare_datasets.md`](datasets/prepare_datasets.md).

You can follow the organization of this dataset to create your own.

### 2. Prepare Dataset Description CSV File
You need to prepare a file that describes the properties of the YUV data in your dataset. We use a CSV file for this purpose, with the following headers: `input_image`, `reference_image`, `yuv_width`, `yuv_height`, `yuv_format`, `yuv_bitdepth`. In a supervised training strategy, `reference_image` will be used as the learning target. In an unsupervised strategy, only `input_image` will be used.
You can refer to [`divkon_2k_yuv_meta_info.csv`](datasets/info/train/divkon_2k_yuv_meta_info.csv) to prepare the description file for your own dataset.

### 3. Generate Dataset Random Split Information PKL File
You need to prepare a dataset split file to define the training, validation, and test sets. You can use our provided script, [`csv_to_random_split_pkl.py`](scripts/csv_to_random_split_pkl.py), to easily accomplish this. The specific command is as follows:
```bash
python3 csv_to_random_split_pkl.py \
  --csv datasets/info/train/divkon_2k_yuv_meta_info.csv \
  --output datasets/info/train/divkon_2k_yuv_5_splits.pkl \
  --splits 5 \
  --seed 42 \
  --train_ratio 0.90 \
  --val_ratio 0.05 \
  --test_ratio 0.05
```
With the command above, the script will create `--splits` random splits based on the information in the input file specified by `--csv`, using the random seed from `--seed` and the ratios defined by `--train/val/test_ratio`. The results will be saved to the path specified by `--output`. We use .pkl files to store the dataset split information.

### 4. Set Train and Validation Dataset Path in Configuration YAML File
After completing the steps above, you are just one step away from training with your own dataset. You need to modify the YAML file under `options/train` to specify the paths for the training and validation datasets. This corresponds to the following information:
```plainText
datasets:
  train:
    name: general_jnd_dataset
    type: GeneralFRDataset

    dataroot_in: ./datasets/divkon_2k_yuv/ori
    dataroot_ref: ./datasets/divkon_2k_yuv/tob_kang_2023_filter_5
    meta_info_file: ./datasets/info/train/divkon_2k_yuv_meta_info.csv
    split_file: ./datasets/info/train/divkon_2k_yuv_5_splits.pkl
    split_index: 1

  val:
    ...
```
`train` and `val` correspond to the datasets used during network training and validation, respectively. `dataroot_in` and `dataroot_ref` are the paths for the input and reference YUV data. `meta_info_file` and `split_file` describe the file properties and dataset splits, while `split_index` selects a specific split for the training run.

### 5. Set Train Parameters
In addition to the dataset information, you can also specify detailed training parameters in the YAML file under `options/train`, including:

Data sampling and augmentation methods:
```plainText
datasets:
  train:
    augment:
      center_crop|random_crop: 224 # Center crop | Random crop size
      hflip: 1 # Enable random horizontal flip
      vflip: 1 # Enable random vertical flip
      rot90: 1 # Enable random 90-degree rotation
      ... # For more methods, refer to transforms.py
```

Batch size during training:
```plainText
datasets:
  train:
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 4 # Dataset enlargement ratio
```

Model parameters for training:
```plainText
network:
  type: MobileIENet_Lite # The type of model to be trained
  ... #  More parameters are determined by the specific model type
```

Training parameter settings:
```plainText
train:
  target: image # Training target, currently only supports predicting the image

  supervised: true # Whether to use a supervised learning strategy

  optim: # Optimizer settings
    type: Adam
    lr: !!float 3e-4
    weight_decay: !!float 1e-6

  scheduler: # Scheduler settings
    type: CosineAnnealingLR
    T_max: 97000
    eta_min: !!float 1e-6
  
  total_iter: 100000 # Total number of training iterations

  warmup_iter: 3000 # Number of warm-up iterations, -1 means no warm-up
```

Loss functions for training:
```plainText
train:
  fidelity_loss_opt: # Fidelity loss, e.g., L1Loss, MSELoss, CharbonnierLoss, Dct8ResidualEnergyLoss
    - type: CharbonnierLoss
      loss_weight: !!float 1.0
    - type: Dct8ResidualEnergyLoss
      loss_weight: !!float 0.02

  rate_loss_opt: # Rate loss, e.g., Dct8HFConstraintLoss
    - type: Dct8HFConstraintLoss
      loss_weight: !!float 0.02

  perceptual_loss_opt: # Perceptual loss, e.g., MsssimLoss, VifLoss
    - type: MsssimLoss
      loss_weight: !!float 0.16
```
For more about loss functions, see the loss function files in the `pyjnd/losses` folder.

### 5. Run Training Command
After preparing the dataset and setting the training parameters, you can use [`train.py`](train.py) .py for a single training run or [`train_nsplits.py`](train_nsplits.py) for multiple training runs.

Single training run command:
```bash
python3 train.py -opt options/train/train_iccv_yan_lite_2025.yml
```

Multiple split training runs command:
```bash
python3 train_nsplits.py -opt options/train/train_iccv_yan_lite_2025.yml
```
>**Note:** For multiple split training, you need to set **split_num** in .yml configuration file greater than 1 
The training results are all located in the `experiments/` folder.

## 📑 Citation

If you find our codes helpful to your research, please consider to use the following citation:

```bib
@misc{pyjnd,
  title={{FJNDF-PyTorch}: JND Filtering Benchmark Platform},
  author={Chenlong He},
  year={2025},
  howpublished = "[Online]. Available: \url{https://github.com/NanUshio/JNDF-Pytorch}"
}
```

## ❤️ Acknowledgement

The code architecture is borrowed from [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) and [BasicSR](https://github.com/xinntao/BasicSR).


## 📧 Contact

If you have any questions, please email `clhe22@m.fudan.edu.cn`