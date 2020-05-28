## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset preparation)
- [Build](#build)
	- [Running](#running)
	- [Configurations](#some important configurations related to RAT)
- [Logging](#logging)




# Introduction

This repo is the Pytorch code for our paper in Ijcai 2020 (Relation-Aware Transformer for Portfolio Policy Learning).

# Dependencies
python 3.7.3 (Anaconda)

pytorch 0.4.1.post2

cudnn 7.4.1

# Dataset preparation
We provide the SP&500 dataset in link https://drive.google.com/drive/folders/1qY-ZWtUxuA_mw9kei-6j5VSwWrck8xbn?usp=sharing. Please download ./database to the same directory as the main.py.

# Build
## Running

cd ${RAT_ROOT}/RAT

./run_main.sh


## Some important configurations related to RAT:


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
In the "train_summary.csv", it saves the net_id, fAPV, SR, CR, TO, St_v and backtest_history.



