## Table of Contents

- [Introduction](#introduction)
	- [Differences from the article version](#differences-from-the-article-version)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Build](#build)
	- [Running](#running)
	- [Configurations](#configurations)
- [Logging](#logging)
- [Citation](#citation)
- [References](#references)




# Introduction

This repo is the Pytorch code for our paper “Relation-Aware Transformer for Portfolio Policy Learning” in IJCAI 2020.

## Differences from the article version  
Note that this library is a part of our main project, and it is several versions ahead of the article.

*In decision-making layer, we construct the leverage operation through two softmax heads, and it's performance is similar to the result of using three softmax heads.

# Dependencies
python 3.7.3 (Anaconda)

pytorch 0.4.1.post2

cudnn 7.4.1

# Dataset
The raw data of Crypto-A is accessed with Poloniex[<sup>[1]</sup>](#refer-anchor-1).  We provide the Crypto-A dataset in link[<sup>[2]</sup>](#refer-anchor-2), where data selection is based on the method  in [<sup>[3]</sup>](#refer-anchor-3). Please download ./database to the same directory as the main.py. The statistics of Crypto-A are summarized as below. 
| Dataset|Assets| Training | Test|
| ---------- | :-----------:  | :-----------: |:-----------: |
|Crypto-A|12|2016.01-2017.11|2017.11-2018.01|



# Build
File main.py mainly contains the construction of RAT network, data preprocessing, the fitting model process and testing process. File run_mian.sh mainly contains the parameter configurations of training RAT.
 
## Running

cd ${RAT_ROOT}/RAT-master

./run_main.sh


## Configurations

The figure shows the entire structure of RAT, and we detail some related parameter configurations in run_main.sh as below.
<img width="500" height="500" src="https://github.com/Ivsxk/RAT/blob/master/RAT_structure.PNG"/>

--x_window_size

    The length of the price series.
    
--local_context_length

    The length of local price context.
    
--daily_interest_rate

    The interest rate of the loan for one day.
    
--log_dir

    The directory to save the log.
    
--model_dir

    The directory to save the model.
    
--model_index

    Set a unique ID for the model.

# Logging
After training process, the model is saved in ${SAVE_MODEL_DIR}/${MODEL_INDEX}.pkl.
After testing process, the backtest results are saved in ${LOG_DIR}/train_summary.csv. It contains metrics such as fAPV, SR, CR and basktest_history.

# Citation
If you use this code for your research, please consider citing:

@inproceedings{xu2020relation,  
  title={Relation-Aware Transformer for Portfolio Policy Learning},  
  author={Xu, Ke and Zhang, Yifan and Ye, Deheng and Zhao, Peilin  and Tan, Mingkui},  
  booktitle={International Joint Conference on Artificial Intelligence},   
  year={2020}  
}

# References
<div id="refer-anchor-1"></div>

- [1] [Poloniex’s official API.](https://poloniex.com/support/api/)

<div id="refer-anchor-2"></div>

- [2] [Crypto-A dataset.](https://drive.google.com/drive/folders/1Icmc5OtTmrLp03JTIdZP849u7ZJ6ytuF)

<div id="refer-anchor-3"></div>

- [3] [Zhengyao Jiang, Dixing Xu, and Jinjun Liang. A deep reinforcement learning framework for the financial port-folio management problem. arXiv, 2017.](https://arxiv.org/abs/1706.10059)
