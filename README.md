# Implementation Codes for HyperDID

## Introduction
This project provides the implementation of the paper "HyperDID: Hyperspectral Intrinsic Image Decomposition with Deep Feature Embedding". [[paper](https://arxiv.org/abs/2311.14899)]

## Requirements

* torch
* cuda

## Project architecture

* demo.py
* data_process.py
* evaluate.py
* model
    * contrastive_learning.py
* data
    * indian
        * IndianPine.mat
        * AVIRIS_colormap.mat
    * pavia
    * houston2013
    * houston2018
* checkpoint
* results

## Running

* Training
> python demo.py --patches 5

* Testing
> python demo.py --patches 5 --flag_test test

## Citing this work

If you find this work helpful for your research, please consider citing:
```
@article{gong2024,
    Author = {Zhiqiang Gong and Xian Zhou and Wen Yao and Xiaohu Zheng and Ping Zhong},
    Title = {HyperDID: Hyperspectral Intrinsic Image Decomposition with Deep Feature Embedding},
    Journal = {IEEE Transactions on Geoscience and Remote Sensing},
    volume = 62,
    Year = {2024}
}
```
