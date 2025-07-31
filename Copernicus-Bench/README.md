# Copernicus-Bench

[![arXiv](https://img.shields.io/badge/arXiv-2503.11849-b31b1b.svg)](https://arxiv.org/abs/2503.11849)
[![License: Code](https://img.shields.io/badge/License--Code-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: data](https://img.shields.io/badge/License--Data-CC--BY--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace Copernicus-Bench](https://img.shields.io/badge/Dataset-Copernicus--Bench-orange?logo=huggingface)](https://huggingface.co/wangyi111/Copernicus-Bench)

This directory contains the official implementation for the evaluation benchmark **Copernicus-Bench** in the paper "Towards a Unified Copernicus Foundation Model for Earth Vision".

## Benchmark Datasets

Copernicus-Bench is a comprehensive evaluation benchmark with 15 hierarchical downstream datasets spread into three level of applications covering all major Sentinel missions (S1,2,3,5P). Among them, 9 are derived from existing datasets, and 6 are newly curated to fill in the gap in ML-ready datasets for S3/5P sensors.

### Dataset characteristics

| Level | Name           | Modality | Task                                | # Images        | Image Size         | # Classes | Source                                                                               | License             |
|-------|----------------|----------|-------------------------------------|-----------------|--------------------|-----------|--------------------------------------------------------------------------------------|---------------------|
| L1    | Cloud-S2       | S2 TOA   | segmentation (cloud)                | 1699/567/551    | 512x512x13         | 4         | [CloudSEN12](https://huggingface.co/datasets/tacofoundation/cloudsen12)              | CC 0 1.0            |
| L1    | Cloud-S3       | S3 OLCI  | segmentation (cloud)                | 1197/399/399    | 256x256x21         | 5         | new                                                                                  | CC BY 4.0           |
| L2    | EuroSAT-S1     | S1 GRD   | classification (LULC)               | 16200/5400/5400 | 64x64x2            | 10        | [EuroSAT-SAR](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR)                 | CC BY 4.0           |
| L2    | EuroSAT-S2     | S2 TOA   | classification (LULC)               | 16200/5400/5400 | 64x64x13           | 10        | [EuroSAT](https://github.com/phelber/EuroSAT)                                        | MIT                 |
| L2    | BigEarthNet-S1 | S1 GRD   | classification (LULC)               | 11894/6117/5991 | 120x120x12         | 19        | [BigEarthNet v2.0](https://bigearth.net/)                                            | CDLA-Permissive-1.0 |
| L2    | BigEarthNet-S2 | S2 SR    | classification (LULC)               | 11894/6117/5991 | 120x120x12         | 19        | [BigEarthNet v2.0](https://bigearth.net/)                                            | CDLA-Permissive-1.0 |
| L2    | LC100Cls-S3    | S3 OLCI  | classification (LULC)               | 5181/1727/1727* | 96x96x21           | 23        | new                                                                                  | CC BY 4.0           |
| L2    | DFC2020-S1     | S1 GRD   | segmentation (LULC)                 | 3156/986/986    | 256x256x13         | 10        | [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) | CC BY 4.0           |
| L2    | DFC2020-S2     | S2 TOA   | segmentation (LULC)                 | 3156/986/986    | 256x256x13         | 10        | [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) | CC BY 4.0           |
| L2    | LC100Seg-S3    | S3 OLCI  | segmentation (LULC)                 | 5181/1727/1727* | 96x96x21 (288x288) | 23        | new                                                                                  | CC BY 4.0           |
| L3    | Flood-S1       | S1 GRD   | change detection (flood)            | 3000/1000/1000  | 224x224x2          | 3         | [Kuro Siwo](https://github.com/Orion-AI-Lab/KuroSiwo)                                | MIT                 |
| L3    | LCZ-S2         | S2 TOA   | classification (local climate zone) | 15000/5000/5000 | 32x32x10           | 17        | [So2Sat LCZ42](https://github.com/zhu-xlab/So2Sat-LCZ42)                             | CC BY 4.0           |
| L3    | Biomass-S3     | S3 OLCI  | regression (biomass)                | 3000/1000/1000* | 96x96x21 (288x288) | 1         | new                                                                                  | CC BY 4.0           |
| L3    | AQ-NO2-S5P     | S5P NO2  | regression (air quality)            | 1480/493/494*   | 56x56x1            | 1         | new                                                                                  | CC BY 4.0           |
| L3    | AQ-O3-S5P      | S5P O3   | regression (air quality)            | 1480/493/494*   | 56x56x1            | 1         | new                                                                                  | CC BY 4.0           |

L1: preprocessing, L2: base applications, L3: specialized applications. *: time series available.

### Dataset access

The benchmark datasets are available on [HuggingFace](https://huggingface.co/datasets/wangyi111/Copernicus-Bench). Optionaly, you can use the [`download_copernicus_bench.sh`](tools/download_copernicus_bench.sh) script to download all the datasets.

```bash
mkdir -p data/copernicusbench
cd data/copernicusbench
bash tools/download_copernicus_bench.sh
```

- [ ] All datasets in Copernicus-Bench will be added to TorchGeo soon.

## Benchmark Implementation

The benchmark codes are adapted from [DOFA-Pytorch](https://github.com/xiong-zhitong/DOFA-pytorch), which mainly depends on the following libraries:

- [PyTorch](https://pytorch.org/) as the deep learning framework.
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for training and logging.
- [Hydra](https://hydra.cc/) for configuration management.
- [TorchGeo](https://github.com/microsoft/torchgeo) for dataset loading.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for segmentation/regression models.
- [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Setup

Install the dependencies:

```bash
conda create -n copernicusbench python=3.10
conda activate copernicusbench
pip install -U openmim
pip install torch==2.1.2
mim install mmcv==2.1.0 mmsegmentation==1.2.2
pip install -e .
```

Set environment variables:

```bash
export MODEL_WEIGHTS_DIR=<path/to/your/where/you/want/to/store/weights>
export DATASETS_DIR=<path/to/your/where/you/want/to/store/datasets>
```

### Running experiments

First, download and extract the Copernicus-Bench datasets to a local directory. 

Then, you can download pretrained foundation model weights to `MODEL_WEIGHTS_DIR`. They will be automatically downloaded later if not found.

Hydra is used for configuration management, with each experiment relying on a model config and a dataset config. For example, you can run the following command to conduct linear probing on the Copernicus-Bench EuroSAT-S2 dataset with Copernicus-FM weights:

```bash
python src/main.py \
output_dir=outputs/cobencheurosats2_copernicusfm_linear \
model=copernicusfm_cls \
dataset=cobench_eurosat_s2 \
lr=0.1 \
task=classification \
num_gpus=0 \
num_workers=8 \
epochs=50 \
warmup_epochs=0 \
seed=42 \
```

You can also use the scripts in [`scripts/`](scripts/) to run multiple datasets/models:

```bash
python scripts/exp_config_copernicusfm.py # frozen encoder evaluation of Copernicus-FM on the whole benchmark
python scripts/exp_config_dofa.py # frozen encoder evaluation of DOFA on the whole benchmark
python scripts/exp_config_vit_sup.py # supervised training from scratch for each dataset in the benchmark
...
```

## Benchmark results

See [`results/`](results/) for the benchmark results. We currently provide frozen-encoder results for the following models:

- Supervised training from scratch (ViT-B/16, ViT-S/16)
- Random initialization (ViT-B/16)
- Copernicus-FM (ViT-B/16)
- DOFA (ViT-B/16)
- CROMA (ViT-B/8, S1/2 only)
- SoftCon (ViT-B/14, S1/2 only)

## License

This directory is licensed under the Apache License 2.0. See each dataset for its specific license.

## Citation

```bibtex
@misc{wang2025unifiedcopernicusfoundationmodel,
      title={Towards a Unified Copernicus Foundation Model for Earth Vision}, 
      author={Yi Wang and Zhitong Xiong and Chenying Liu and Adam J. Stewart and Thomas Dujardin and Nikolaos Ioannis Bountos and Angelos Zavras and Franziska Gerken and Ioannis Papoutsis and Laura Leal-Taixé and Xiao Xiang Zhu},
      year={2025},
      eprint={2503.11849},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11849}, 
}
```
