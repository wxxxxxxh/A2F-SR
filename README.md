# Lightweight Single-Image Super-Resolution Network with Attentive Auxiliary Feature Learning

A official implementation about our paper: Lightweight Single-Image Super-Resolution Network with Attentive Auxiliary Feature Learning(ACCV2020)

[Paper(Arxiv)](https://arxiv.org/pdf/2011.06773.pdf)

[Project Page](https://cv.wangxuehui.site/SR/)

## Dependecies
- python = 3.6
- pytorch = 1.1.0 (higher is OK)
- torchvision = 0.3.0
- cudatoolkit = 9.0
- tqdm = 4.48.0
- easydict = 1.9
- scikit-image = 0.17.2
- opencv-contrib-python = 4.3.0.36
- pyyaml = 5.3.1
- pillow = 6.2.2
- scipy = 1.2.1

If you use Anaconda, you can run these commands to build a env that can run our code correctly:
```bash
#First step: build a env and enter it
conda create -n AAF python=3.6
conda activate AAF

#Second step: install pytorch and torchvision via conda
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

#Final step: install other dependecies via pip
pip install -r requirements.txt
```
Now, you have build a conda env for running code successfully!

## Training
Firstly, please refer to [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to get DIV2K which is a widely-used dataset for super-resolution. Put it in any folder, and modify the parameter `dataset.dir_root` in yaml(e.g. `yaml/train_sd_x4.yaml`) as this folder. Our method will scan files according to `dataset.dir_root/dataset.dir_lr`.

Perform below script to train the model from scratch.
```bash
python train.py --config AAF_SD_x4  #AAF-SD for x4 scale
```
Here, AAF_SD_x4 can be change as:
```
AAF_S_x4    #AAF-S for x4 scale
AAF_SD_x4   #AAF-SD for x4 scale
AAF_M_x4    #AAF-M for x4 scale
AAF_L_x4    #AAF-L for x4 scale
AAF_S_x3    #AAF-S for x3 scale
AAF_SD_x3   #AAF-SD for x3 scale
AAF_M_x3    #AAF-M for x3 scale
AAF_L_x3    #AAF-L for x3 scale
AAF_S_x2*   #AAF-S for x2 scale
AAF_SD_x2   #AAF-SD for x2 scale
AAF_M_x2*   #AAF-M for x2 scale
AAF_L_x2*   #AAF-L for x2 scale
```


## Testing
Please put the testing data (such like `DIV2K_test_LR_bicubic/X4`) that you want to check results in `./test_img`, now you will have:

```
|- ./test_img/
|--- DIV2K_test_LR_bicubic/
|------ X4/
|--------- 0901x4.png
|--------- 0902x4.png
|--------- ...
```

We provide the pretrained model of AAF in `./checkpoint`, so you can test images directly. Run this commands to obatain the results. They can be seen in `./results/AAF_SD_x4`:

```bash
python test.py --config AAF_SD_x4
```

## Others
#### Paramaters
- Parameters used in our code can be seen in `./yaml/*.yaml`.

#### *About pretrained model
- Unfortunately, we encounter a problem with our hard disk, which cause a loss of data including the pretrained model (x2 settings except AAF-SD). We will release the other x2 checkpoints later.

## Citation
If you think the paper is helpful for your research, please cite:
```
@inproceedings{wang2020Lightweight,
        author = {Xuehui Wang, Qing Wang, Yuzhi Zhao, Junchi Yan, Lei Fan, and Long Chen.},
        title = {Lightweight Single-Image Super-Resolution Network with Attentive Auxiliary Feature Learning},
        journal = {ACCV},
        year = {2020}
}
```

If you have any question, please contact wangxh228@mail2.sysu.edu.cn

