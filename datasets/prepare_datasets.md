# Test Dataset Description

Our algorithm focuses on perceptual coding optimization for natural content. We evaluate its performance on commonly used natural‐content datasets, specifically:

1. **HEVC SDR Common Test Conditions (CTC)** – Class B sequences at 1920×1080 (YUV420p) [1]
2. **XIPH Dataset** – all 1080p YUV420p sequences [2]
3. **MCL-JCV Dataset** – all 1080p YUV420p sequences [3]  
4. **MCL-JCI Dataset** – all 1080p YUV420p sequences [4]

> **Note:** We only test videos at 1080p resolution from each dataset. If you use any of the above open datasets, please cite the corresponding sources: HEVC SDR CTC [1], XIPH [2], MCL-JCV [3], and MCL-JCI [4].
---

## HEVC SDR CTC (1080p)

We provide a consolidated archive of all Class B (1920×1080) SDR test sequences in YUV420p format:

- **Archive name:** `hevc_sdr_ctc.tar.gz`  
- **Download (Baidu Netdisk Super Member v6):**  
  - **URL:** [https://pan.baidu.com/s/1X9zkxCE83nK7Tgp9nWXMsQ?pwd=1224](https://pan.baidu.com/s/1X9zkxCE83nK7Tgp9nWXMsQ?pwd=1224)  
  - **Extraction code:** `1224`  

### Extraction Example
```bash
# Download the HEVC SDR CTC archive
# Upload .tar.gz file to datasets/hevc_sdr_ctc
cd datasets
mkdir hevc_sdr_ctc


# Extract all YUV files
tar -xvzf hevc_sdr_ctc.tar.gz
cd ori

# Listing files
ls
# Kimono.yuv
# ParkScene.yuv
# Cactus.yuv
# BasketballDrive.yuv
# BQTerrace.yuv
````

---

## XIPH Dataset (1080p)

We provide a consolidated archive of all 1080p YUV420p sequences from the XIPH dataset:

* **Archive name:** `xiph_dataset_1080p.tar.gz`
* **Download (Baidu Netdisk Super Member v6):**

  * **URL:** [https://pan.baidu.com/s/1WR7t01e5pNW5oCs5_S9XAg?pwd=1224](https://pan.baidu.com/s/1WR7t01e5pNW5oCs5_S9XAg?pwd=1224)
  * **Extraction code:** `1224`


### Extraction Example

```bash
# Download the XIPH dataset archive
# Upload .tar.gz file to datasets/xiph_dataset_1080p
cd datasets
mkdir xiph_dataset_1080p

# Extract all YUV files
tar -xvzf xiph_dataset_1080p.tar.gz
cd ori

# Listing files
ls
# blue_sky_1080p25.yuv
# crowd_run_1080p50.yuv
# dinner_1080p30.yuv
# ducks_take_off_1080p50.yuv
# factory_1080p30.yuv
# in_to_tree_1080p50.yuv
# life_1080p30.yuv
# old_town_cross_1080p50.yuv
# park_joy_1080p50.yuv
# pedestrian_area_1080p25.yuv
# riverbed_1080p25.yuv
# rush_hour_1080p25.yuv
# station2_1080p25.yuv
# sunflower_1080p25.yuv
# tractor_1080p25.yuv
```

## MCL-JCV Dataset (1080p)

We provide a consolidated archive of all 1080p YUV420p sequences from the MCL-JCV dataset:

* **Archive name:** `mcl_jcv_dataset_1080p.tar.gz`
* **Download (Baidu Netdisk Super Member v6):**

  * **URL:** [https://pan.baidu.com/s/18b2MoYfWjiWM_IG7rUcb2w?pwd=1224](https://pan.baidu.com/s/18b2MoYfWjiWM_IG7rUcb2w?pwd=1224)
  * **Extraction code:** `1224`

### Extraction Example

```bash
# Download the MCL-JCV dataset archive
# Upload .tar.gz file to datasets/mcl_jcv_dataset_1080p
cd datasets
mkdir mcl_jcv_dataset_1080p

# Extract all YUV files
tar -xvzf mcl_jcv_dataset_1080p.tar.gz
cd ori

# Listing files
ls
# videoSRC01_1920x1080_30.yuv
# videoSRC02_1920x1080_30.yuv
# videoSRC03_1920x1080_30.yuv
# videoSRC04_1920x1080_30.yuv
# videoSRC05_1920x1080_25.yuv
# videoSRC06_1920x1080_25.yuv
# videoSRC07_1920x1080_25.yuv
# videoSRC08_1920x1080_25.yuv
# videoSRC09_1920x1080_25.yuv
# videoSRC10_1920x1080_30.yuv
# ...
```

## MCL-JCI Dataset (1080p)

We provide a consolidated archive of all 1080p YUV420p images from the MCL-JCI dataset:

* **Archive name:** `mcl_jci_dataset_1080p.tar.gz`
* **Download (Baidu Netdisk Super Member v6):**

  * **URL:** [https://pan.baidu.com/s/1x2e081YNuxgRpRy2jsIBVA?pwd=1224](https://pan.baidu.com/s/1x2e081YNuxgRpRy2jsIBVA?pwd=1224)
  * **Extraction code:** `1224`


### Extraction Example

```bash
# Download the MCL-JCI dataset archive
# Upload .tar.gz file to datasets/mcl_jci_dataset_1080p
cd datasets
mkdir mcl_jci_dataset_1080p

# Extract all YUV files
tar -xvzf mcl_jci_dataset_1080p.tar.gz
cd ori

# Listing files
ls
# ImageJND_SRC01.yuv
# ImageJND_SRC02.yuv
# ImageJND_SRC03.yuv
# ImageJND_SRC04.yuv
# ImageJND_SRC05.yuv
# ImageJND_SRC06.yuv
# ImageJND_SRC07.yuv
# ImageJND_SRC08.yuv
# ...
```

# Train Dataset Description

For training purposes, we provide our custom divkon_2k dataset. This dataset is a combination of two popular high-resolution datasets: DIV2K and KonIQ-10k. We have made it available for download via Baidu Netdisk.

* **Archive name:** `divkon_2k_yuv.tar.gz`
* **Download (Baidu Netdisk Super Member v6):**

  * **URL:** [https://pan.baidu.com/s/1cOr8u1ZVCUcbmr4eGwzdVg?pwd=1224](https://pan.baidu.com/s/1cOr8u1ZVCUcbmr4eGwzdVg?pwd=1224)
  * **Extraction code:** `1224`

The dataset archive contains multiple versions of the images, which are suitable for various supervised learning strategies:

> **Note**: If you use any of the above open datasets, please cite the corresponding sources: DIV2K [5], KonJND-1K [6].

* ori: Contains the original YUV files.
* tip_bae_2016_filter_4: Contains YUV files processed with a specific JND-Guided pre-filter, which can serve as a reference method for training networks.

### Extraction Example
```bash
# Download the divkon_2k dataset archive
# Upload .tar.gz file to datasets/divkon_2k_yuv
cd datasets
mkdir divkon_2k_yuv

# Extract all YUV files
tar -xzf divkon_2k_yuv.tar.gz

# Listing dirs
tree -d
# .
# ├── ori
# └── tip_bae_2016_filter_4

cd ori
# Listing files
ls
# ...
# SRC0001.yuv
# SRC0002.yuv
# ...
```


## References
[1] Frank Bossen. “Common test conditions and software reference configurations.” *3rd JCT-VC Meeting*, Guangzhou, China, Oct. 2010.

[2] Xiph.Org Foundation. “Xiph.org Video Test Media (derf’s collection).” 2013. Available at: https://media.xiph.org/video/derf/ (Accessed: 2025-09-30).  

[3] Haiqiang Wang, Weihao Gan, Sudeng Hu, Joe Yuchieh Lin, Lina Jin, Longguang Song, Ping Wang, Ioannis Katsavounidis, Anne Aaron, and C.-C. Jay Kuo. “MCL-JCV: a JND-based H.264/AVC video quality assessment dataset.” *Proc. IEEE ICIP*, pp. 1509–1513, 2016.  

[4] Lina Jin, Joe Yuchieh Lin, Sudeng Hu, Haiqiang Wang, Ping Wang, Ioannis Katsavounidis, Anne Aaron, and C.-C. Jay Kuo. “Statistical study on perceived JPEG image quality via MCL-JCI dataset construction and analysis.” *Electronic Imaging*, 2016(13):1–9, 2016.

[5] Eirikur Agustsson and Radu Timofte. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 126–135, 2017.

[6] Hanhe Lin, Guangan Chen, Mohsen Jenadeleh, Vlad Hosu, Ulf-Dietrich Reips, Raouf Hamzaoui, and Dietmar Saupe. “Large-Scale Crowdsourced Subjective Assessment of Picturewise Just Noticeable Difference.” *IEEE Transactions on Circuits and Systems for Video Technology*, 2022.