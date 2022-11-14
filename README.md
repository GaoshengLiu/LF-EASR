# LF-EASR: Efficient Light Field Angular Super-Resolution With Sub-Aperture Feature Learning and Macro-Pixel Upsampling

This repository contains official pytorch implementation of Efficient Light Field Angular Super-Resolution With Sub-Aperture Feature Learning and Macro-Pixel Upsampling in TMM 2022, by Gaosheng Liu, Huanjing Yue, Jiamin Wu, and Jingyu Yang. [TMM 2022 LF-EASR](https://ieeexplore.ieee.org/document/9915519)

##Requirement
* Ubuntu 18.04
* Python 3.6
* Pyorch 1.7
* Matlab

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders.

## Training
* Run:
  ```python
  python train.py
## Test
* Run:
  ```python
  python test.py

## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{liu2022efficient,
  title={Efficient Light Field Angular Super-Resolution With Sub-Aperture Feature Learning and Macro-Pixel Upsampling},
  author={Liu, Gaosheng and Yue, Huanjing and Wu, Jiamin and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
## Acknowledgement
Our work and implementations are based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!
