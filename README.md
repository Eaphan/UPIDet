<!-- <img src="docs/open_mmlab.png" align="right" width="30%"> -->

# Unleash the Potential of Image Branch for Cross-modal 3D Object Detection
This is the official implementation of "Unleash the Potential of Image Branch for Cross-modal 3D Object Detection". This repository is based on [`[OpenPCDet]`](https://github.com/open-mmlab/OpenPCDet).

<!-- <img src="docs/pipeline.png"> -->
**Abstract**: To achieve reliable and precise scene understanding, autonomous vehicles typically incorporate multiple sensing modalities to capitalize on their complementary attributes. However, existing cross-modal 3D detectors do not fully utilize the image domain information to address the bottleneck issues of the LiDAR-based detectors. This paper presents a new cross-modal 3D object detector, namely UPIDet, which aims to unleash the potential of the image branch from two aspects. First, UPIDet introduces a new 2D auxiliary task called normalized local coordinate map estimation. This approach enables the learning of local spatial-aware features from the image modality to supplement sparse point clouds. Second, we discover that the representational capability of the point cloud backbone can be enhanced through the gradients backpropagated from the training objectives of the image branch, utilizing a succinct and effective point-to-pixel module. Extensive experiments and ablation studies validate the effectiveness of our method. Notably, we achieved the top rank in the highly competitive cyclist class of the KITTI benchmark at the time of submission. 



## Overview
- [Installation](#Installation)
- [Pretrained Models](#pretrained-models)
- [Getting Started](#getting-started)
- [License](#license)
- [Acknowledgement](#acknowledgement)
<!-- - [Contribution](#contribution) -->
<!-- - [Citation](#citation) -->

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation instruction.

## Pretrained-models
Here we present the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.

* The pre-trained model is trained with 4 NVIDIA 3090Ti GPUs and are available for download.
* The training time is measured with 4 NVIDIA 3090Ti GPUs and PyTorch 1.8.
* We could not provide the above pretrained models due to Waymo Dataset License Agreement, but you could easily achieve similar performance by training with the default configs.

|                                             | training time | Car@R40 | Pedestrian@R40 | Cyclist@R40   | download |
|---------------------------------------------|:----------:|:-------:|:-------:|:-------:|:---------:|
| [UPIDet](tools/cfgs/kitti_models/upidet.yaml) |~12 hours| 86.10 | 68.67 | 76.70 | [model-287M](https://drive.google.com/file/d/1clUCPAixSAAad5aSH08zJr32-8o--P0u/view?usp=sharing) |

## Getting Started

### Prepare KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
<!-- * If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI training set -->
<!-- * NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged. -->

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & planes
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
Especially, for the 2D auxiliary task of semantic segmentation, we used the instance segmentation annotations as provided in [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset). We incorporate mask of instance segmentation in kitti_infos_train/val.pkl and kitti_dbinfos_train.pkl. Please download them in this [link](https://drive.google.com/drive/folders/1cyFt9MqHnKK620IKbRuTN6SiEvJP6r8d?usp=sharing) and replace the original files.

### Prepare Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/)(v1.2.0), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
GLENet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── kitti_format
│   │   │   │── calib
│   │   │   │── image_0
│   │   │   │── image_1
│   │   │   │── image_2
│   │   │   │── image_3
|   |   |   |── image_4
|   |   |   |   │── segment-xxxxxxxx_with_camera_labels
|   |   |   |   |   │── 0000.jpg  0001.jpg  0002.jpg ...
|   |   |   |   |── ...
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)

```
* You should use mmdet3d to generate RGB images for waymo dataset. Then you can link the image files to  the kitti_format directory using modified script tools/map_mmdet_waymo_image.py.

* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

### Training
```
cd tools;
python train.py --cfg_file ./cfgs/kitti_models/upidet.yaml
```
Multi gpu training, assuming you have 4 gpus:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --cfg_file ./cfgs/kitti_models/upidet.yaml

```

**Note**: For the waymo dataset, you should checkout branch "waymo_lidar" to train the single-modal detector, then checkout branch "waymo" to train the cross-modal detector based on the weights of obtained single-modal detector.

### Testing
```
cd tools/
```
Single gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
python test.py --eval_all --cfg_file ./cfgs/kitti_models/upidet.yaml
```

Multi gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh 4 --eval_all --cfg_file ./cfgs/kitti_models/upidet.yaml
```

Multi gpu testing a specific checkpoint, assuming you have 4 gpus and checkpoint_39 is your best checkpoint :
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh 4  --cfg_file ./cfgs/kitti_models/upidet.yaml --ckpt ../output/upidet/default/ckpt/checkpoint_epoch_80.pth
```

<!-- ## Pretrained Models -->

## License

`UPIDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes:
- [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [Focalsconv](https://github.com/dvlab-research/FocalsConv)
- [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [PDV](https://github.com/TRAILab/PDV)

## Citation
If you find this work useful in your research, please consider cite:
```
@inproceedings{zhang2024unleash,
    title={Unleash the potential of image branch for cross-modal 3d object detection},
    author={Zhang, Yifan and Zhang, Qijian and Hou, Junhui and Yuan, Yixuan and Xing, Guoliang},
    booktitle={Advances in Neural Information Processing Systems},
    volume={36},
    pages={51562--51583},
    year={2023}
}
```
