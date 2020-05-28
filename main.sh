#!/bin/bash
set -e
set -x
export LANG=en_US.UTF-8
#export CUDA_VISIBLE_DEVICES=5
pip install pandas

disk=/opt/ml/disk
ROOT=${disk}/pgportfolio
if [ ! -d "${ROOT}" ];then
  mkdir ${ROOT}
else
  echo "${ROOT} exists"
fi

WORKSPACE=${ROOT}/transformer_no_long_term_sigmoid_cov_attention_pe_no_sigmoid_local_attention_linear_decoder_asset_seven_short_sale
if [ ! -d "${WORKSPACE}" ];then
  mkdir ${WORKSPACE}
else
  echo "${WORKSPACE} exists"
fi

LOG_DIR=${WORKSPACE}/log
if [ ! -d "$LOG_DIR" ]; then
  echo "Creating ${LOG_DIR} ..."
  mkdir -p ${LOG_DIR}
fi

MODEL_DIR=${WORKSPACE}/model
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating ${MODEL_DIR} ..."
  mkdir -p ${MODEL_DIR}
fi


for idx in 1 2 3 4 5 6;
do
for xws in 31;
do
for yws in 0;
do
for bz in 128;
do
for mh in 2;
do
for fd in 5;
do
for md in 12;
do
for wd in 5e-7;
do
for dir in 0.001;
do
MODEL_NAME=transformer

INFOR_DIR=${LOG_DIR}/${MODEL_NAME}_xws_${xws}_yws_${yws}_bz_${bz}_mh_${mh}_fd_${fd}_md_${md}_wd_${wd}_dir_${dir}
SAVE_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}_xws_${xws}_yws_${yws}_bz_${bz}_mh_${mh}_fd_${fd}_md_${md}_wd_${wd}_dir_${dir}

if [ ! -d "$INFOR_DIR" ]; then
  echo "Creating ${INFOR_DIR} ..."
  mkdir -p ${INFOR_DIR}
fi

if [ ! -d "$SAVE_MODEL_DIR" ]; then
  echo "Creating ${SAVE_MODEL_DIR} ..."
  mkdir -p ${SAVE_MODEL_DIR}
fi

python main.py \
--total_step 50000 \
--x_window_size ${xws} \
--y_window_size ${yws} \
--batch_size ${bz} \
--coin_num 11 \
--feature_number 4 \
--output_step 1000 \
--test_portion 0.08 \
--trading_consumption 0.0025 \
--variance_penalty 0.0 \
--cost_penalty 0.0 \
--learning_rate 0.0001 \
--start 2016/01/01 \
--end 2018/01/01 \
--model_dir ${SAVE_MODEL_DIR} \
--model_name ${MODEL_NAME} \
--model_index ${idx} \
--model_dim ${md} \
--multihead_num ${mh} \
--filter_d ${fd} \
--weight_decay ${wd} \
--daily_interest_rate ${dir} \
--log_dir ${INFOR_DIR} > ${INFOR_DIR}/${idx}_info.txt 2>&1 
done
done
done
done
done
done
done
done
done

