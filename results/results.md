# Results Directory Structure

The `results` directory is organized by video dataset and GOP structure, and contains performance results for various filtering methods across different encoders. Below is the top-level layout:

```
results/
├── hevc_sdr_ctc
│   ├── result.md
│   └── test_condition
├── mcl_jci_dataset_1080p
│   ├── result.md
│   └── test_condition
├── mcl_jcv_dataset_1080p
│   ├── result.md
│   └── test_condition
├── parse_csv_to_md.py
├── results.md
└── xiph_dataset_1080p
    ├── result.md
    └── test_condition
```

* **Dataset folders** (e.g., `hevc_sdr_ctc`): each folder corresponds to a specific test collection of video sequences.
* **Markdown files** (`result.md`): record the average compression performance of different JND filters.
* **CSV files** (`<filter_method>_<encoder>.csv`): contain BD‑Rate and BD‑PSNR results for each filter method applied to a given encoder. Columns are:

  * `video`: sequence name; `ALL` denotes the average over all test videos.
  * `metric`: quality metric evaluated (e.g., `psnr`, `ms_ssim`, `vmaf`, etc.).
  * `BD-Rate(%)`: percentage change in BD‑Rate compared to the anchor.
  * `BD-PSNR(dB)`: corresponding BD‑PSNR delta.
* **YAML config files** (`*.yml`): each encoder has a YAML file (e.g., `x264.yml`) specifying the encoding parameters used.
* **Regression script** (`regr_encode.sh`): batch script to run all encoder–filter combinations and generate the CSV results.
* **Optional `test_condition` folders**: contain alternative `.yml` configs and the same `regr_encode.sh` for specialized test conditions.

---

For detailed results, please refer to 
* [`hevc_sdr_ctc/result.md`](hevc_sdr_ctc/result.md)
* [`xiph_dataset_1080p/result.md`](xiph_dataset_1080p/result.md)
* [`mcl_jcv_dataset_1080p/result.md`](mcl_jcv_dataset_1080p/result.md)
* [`mcl_jci_dataset_1080p/result.md`](mcl_jci_dataset_1080p/result.md)

For detailed results of each video, you could download corresponding csv file from
  * **URL:** [https://pan.baidu.com/s/1atA9BbX6FNoQDIiuxGRIXQ?pwd=1224](https://pan.baidu.com/s/1atA9BbX6FNoQDIiuxGRIXQ?pwd=1224)
  * **Extraction code:** `1224`