# An Efficient Nuclear Instance Segmentation Approach

The is the final project of COMP576. 



## Installation
This repo is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [INSTALL.md](INSTALL.md) for installation instructions.


## Training and Testing
**Train(4gpu):**
### CFM+HBB
```
bash ./tools/dist_train.sh configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10.py 4 --launcher pytorch --work_dir work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10
```
### HBB
```
CUDA_VISIBLE_DEVICES=2,3,4,6 bash ./tools/dist_train.sh configs/panNuke/polar_init_refine_r50_centerness_polar.py 4 --launcher pytorch --work_dir work_dirs/panNuke/polar_init_refine_r50_centerness_polar
```
### PolarMask
```
CUDA_VISIBLE_DEVICES=2,3,4,6 bash ./tools/dist_train.sh configs/polarmask/4gpu/polar_init_r50_centerness_polar.py 4 --launcher pytorch --work_dir work_dirs/panNuke/polar_init_r50_centerness_polar
```


**Test(4gpu):**
### CFM+HBB
```
bash tools/dist_test.sh configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10/latest.pth 4 --out work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10/res.pkl --eval segm
```
### HBB
```
bash tools/dist_test.sh configs/panNuke/polar_init_refine_r50_centerness_polar.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar/latest.pth 4 --out ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar/res.pkl --eval segm
```
### PolarMask
```
bash tools/dist_test.sh configs/panNuke/polar_init_r50_centerness_polar.py ./work_dirs/panNuke/polar_init_r50_centerness_polar/latest.pth 4 --out ./work_dirs/panNuke/polar_init_r50_centerness_polar/res.pkl --eval segm
```



## Acknowledgements

We gratefully acknowledge the contributions of the open-source repos, like [mmdetection](https://github.com/open-mmlab/mmdetection) and [PolarMask](https://github.com/xieenze/PolarMask).  

This work utilized the PanNuke dataset and was built using the PyTorch framework. We thank all prior researchers and developers whose work has inspired this project.
