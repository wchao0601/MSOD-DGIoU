<div align="center">
<!-- <h1> LBGD </h1> -->
<h2> <a href="https://ieeexplore.ieee.org/document/11150448">MSOD: A Large-Scale Multi-Scene Dataset and a Novel Diagonal-Geometry Loss for SAR Object Detection</h2>
<h3> Chao Wang, Wenxuan Fang, Xiang Li, Jian Yang, Lei Luo*</h3>
<h4> 2025</h4>
</div>

#### This repository contains the official implementation of the paper: MSOD: A Large-Scale Multi-Scene Dataset and a Novel Diagonal-Geometry Loss for SAR Object Detection, TGRS 2025

## ✨ Overview
Synthetic Aperture Radar (SAR) has attracted significant attention due to its excellent all-weather imaging capabilities. However, SAR image object detection methods face two major challenges: 1) Most existing datasets are small in volume and single in category and scene. 2) Existing IoU-based loss functions cannot fully capture the relationship between prediction and target bounding boxes. To further advance the development of the SAR object detection method, we construct a large-scale multi-scene SAR object detection dataset called MSOD. It comprises three distinct scenarios, containing 40K images and about 1M instances of interest classified into six categories. In addition, we propose a novel diagonal-based similarity loss, Diagonal-Geometry IoU (DGIoU), to optimize the performance of SAR object detection by measuring the similarity between the diagonal of the prediction and target boxes. Specifically, we equivalently represent a rectangular box as a diagonal, and then define DGIoU based on the similarity of a set of sampling points between the diagonals of the predicted box and the target box. DGIoU effectively characterizes the difference between the predicted box and the target box, particularly in box inclusion and separation cases, resulting in improved localization accuracy. Numerous experimental results demonstrate that MSOD is closer to practical application and more challenging than existing SAR image datasets, and serves as a strong benchmark for evaluating the effectiveness of various IoU loss functions.


<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/overall-network.png" width="99.5%"> </p>



## 📄 Documentation
### Installation
Create and activate a conda environment:
```
conda create -n lbgd python=3.11
conda activate lbgd
```
Install the required packages:
```
git clone https://github.com/wchao0601/LBGD.git
cd LBGD/
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
pip install -r requirements.txt
```


### Data Preparation
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/data-stastics1.png" width="99.5%"> </p>
| Dataset | Down-Link | Image Size |
| :---: | :---: | :---: |
| MSOD | [Baidu](https://pan.baidu.com/s/1bVY9rd9Q_XRLAqgIM615Ow?pwd=0601)|1024 x 1024|


### Train
```python
python train.py
```

### Test
```python
python test.py
```

### Predict
```python
python predict.py
```

## 📈 Results
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/results1.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/results2.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/vis-detection.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/vis-heatmap.png" width="99.5%"> </p>

|  Model     | YOLOv8S | YOLOv8N | Mimic | CWD | MGD | PKD | CrossKD | LSKD | LBGD (Ours)|
| :---:      | :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Weights   |[Download](https://pan.baidu.com/s/1wLSLH8xxIRUmrbhCGv59bQ?pwd=0601)|[Download](https://pan.baidu.com/s/1yHA6ER-iT2gV-KMYh9XKjQ?pwd=0601)|[Download](https://pan.baidu.com/s/1s10Dgkc4AHA0ERQI0MynNg?pwd=0601)|[Download](https://pan.baidu.com/s/1gl8I_uJENIfLNPCIBMWmIg?pwd=0601)|[Download](https://pan.baidu.com/s/1Iz16XkU_JB8PBKTMfxWnSA?pwd=0601)|[Download](https://pan.baidu.com/s/1B4gMxwclgogyCosHkYla3A?pwd=0601)|[Download](https://pan.baidu.com/s/1_UvWYoRsh15UYMBkWJYeAg?pwd=0601)|[Download](https://pan.baidu.com/s/18w-6Z23H40oRO_uCRgIXhQ?pwd=0601)|[Download](https://pan.baidu.com/s/16aKLoscwT10lvOEXE0Pvuw?pwd=0601)|

## 🌐 Contact
If you have any questions, please feel free to contact me via email at wchao0601@163.com

## 📚 Citation
If our work is helpful, you can cite our paper:
```
@article{wang2025msod,
  title={MSOD: A Large-Scale Multi-Scene Dataset and a Novel Diagonal-Geometry Loss for SAR Object Detection},
  author={Wang, Chao and Fang, Wenxuan and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}

@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
}

@article{wang2023category,
  title={Category-oriented localization distillation for sar object detection and a unified benchmark},
  author={Wang, Chao and Ruan, Rui and Zhao, Zhicheng and Li, Chenglong and Tang, Jin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--14},
  year={2023},
  publisher={IEEE}
}

```
## 🙏 Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics), which is excellent works.

