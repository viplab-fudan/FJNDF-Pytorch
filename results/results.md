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

In our benchmark, we have conducted a comprehensive evaluation of all combinations of the JND modeling and injection methods. Each combination is named using the convention: `JND_MODELING_METHOD_filter_INJECTION_METHOD`. For instance, `spl_bae_2013_filter_2` utilizes the `spl_bae_2013` JND modeling method with the injection strategy corresponding to `target=2`. Detailed descriptions of each modeling and injection method can be found in [models.md](../pyjnd/models.md).

It is worth noting that:
- `tob_kang_2023_filter_3` corresponds to the JND filtering method proposed in [1].
- `tip_bae_2016_filter_4` corresponds to the JND filtering method proposed in [2].

The current results are exclusively for **image coding** applications. Consequently, any temporal methods described in [1] and [2] have not been utilized. Furthermore, our benchmark focuses specifically on the JND filtering process itself. As a result, other aspects such as detail enhancement for salient regions (from [1] and [2]) and encoder QP adjustments (from [2]) are not considered in this evaluation.

In addition to these traditional methods, the results also include the performance of several neural network-based approaches:
- **`ojcas_sun_2024`**: The lightweight JND network proposed in [3].
- **`iccv_yan_2025`**: The lightweight image enhancement network from [4].
- **`jccv_yan_lite_2025`**: Our proposed network, which is a specialized and lightweight version of the architecture from [4], tailored for the JND filtering task. This model corresponds to the network in our ISCAS 2026 submission.

Crucially, for the methods from [3] and [4], we only adopted their network architectures. The training for **all** neural network models was unified under the **JND-guided supervised training framework** proposed in our article submitted to ISCAS 2026.

---

For detailed results, please refer to 
* [`hevc_sdr_ctc/result.md`](hevc_sdr_ctc/result.md)
* [`xiph_dataset_1080p/result.md`](xiph_dataset_1080p/result.md)
* [`mcl_jcv_dataset_1080p/result.md`](mcl_jcv_dataset_1080p/result.md)
* [`mcl_jci_dataset_1080p/result.md`](mcl_jci_dataset_1080p/result.md)

For detailed results of each video, you could download corresponding csv file from
  * **URL:** [https://pan.baidu.com/s/1itaDTTeSkgUtnzcZWefCNA?pwd=1224](https://pan.baidu.com/s/1itaDTTeSkgUtnzcZWefCNA?pwd=1224)
  * **Extraction code:** `1224`

---

### 🚀 Model Complexity and Performance

We provide a detailed comparison of model size, computational cost (GFLOPs), and inference latency for our proposed method against several baselines. The results validate that our network design achieves a state-of-the-art balance between efficiency and effectiveness.

| Method | Parameters (K) (Train) | Parameters (K) (Inference) | GFLOPs | CPU Latency (s) | GPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| tob_kang_2023_filter_3 [1] | / | / | / | ~70 | / |
| tip_bae_2016_filter_4 [2] | / | / | / | ~20 | / |
| ojcas_sun_2024 [3] | 12.23 | 12.22 | 50.70 | 0.835 | 15.43 |
| iccv_yan_2025 [4] | 38.92 | 3.06 | 12.15 | 0.595 | 13.04 |
| iccv_yan_lite_2025 (**Ours**) | 24.95 | **1.86** | **7.15** | **0.428** | **10.88** |

**Note:** All latency measurements were performed on the following hardware platforms:
*   **CPU**: Intel(R) Xeon(R) Gold 6230 CPU
*   **GPU**: NVIDIA GeForce RTX 2080 Ti GPU

---

## References

[1] Byeongkeun Kang and Wonha Kim. “Human Perception-oriented Enhancement and Smoothing for Perceptual Video Coding.” IEEE Transactions on Broadcasting, 2023.

[2] Huajie Tan, Guoqing Xiang, Xiaodong Xie, and Huizhu Jia. “Joint Frame-level and Block-level Rate-Perception Optimized Preprocessing for Video Coding.” Proceedings of the 6th ACM International Conference on Multimedia in Asia, 2024.

[3] Sun, Yu-Han, Lee, Chiang Lo-Hsuan, and Chang, Tian-Sheuan, "IQNet: Image quality assessment guided just noticeable difference prefiltering for versatile video coding," IEEE Open Journal of Circuits and Systems, vol. 5, pp. 17-27, 2023.

[4] Yan, Hailong and Li, Ao and Zhang, Xiangtao and Liu, Zhe and Shi, Zenglin and Zhu, Ce and Zhang, Le, "MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices," arXiv preprint arXiv:2507.01838, 2025.