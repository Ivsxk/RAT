#!/bin/bash
set -e
set -x
export LANG=en_US.UTF-8
#export CUDA_VISIBLE_DEVICES=5
pip install pandas



LOG_DIR=${HOME}/PGPortfolio-master/PGPortfolio-master/pgportfolio/RAT/log
if [ ! -d "$LOG_DIR" ]; then
  echo "Creating ${LOG_DIR} ..."
  mkdir -p ${LOG_DIR}
fi

MODEL_DIR=${HOME}/PGPortfolio-master/PGPortfolio-master/pgportfolio/RAT/model
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating ${MODEL_DIR} ..."
  mkdir -p ${MODEL_DIR}
fi



for MODEL_INDEX in 1 2 3 4 5;
do
for X_WINDOW_SIZE in 31;
do
for BATCH_SIZE in 128;
do
for MULTIHEAD_NUM in 2;
do
for LOCAL_CONTEXT_LENGTH in 5;
do
for MODEL_DIM in 12;
do
for WEIGHT_DECAY in 5e-8;
do
for DAILY_INTEREST_RATE in 0.001;
do
MODEL_NAME=RAT

INFOR_DIR=${LOG_DIR}/${MODEL_NAME}_xws_${X_WINDOW_SIZE}_yws_${yws}_bz_${BATCH_SIZE}_mh_${MULTIHEAD_NUM}_lcl_${LOCAL_CONTEXT_LENGTH}_md_${MODEL_DIM}_wd_${WEIGHT_DECAY}_dir_${DAILY_INTEREST_RATE}
SAVE_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}_xws_${X_WINDOW_SIZE}_yws_${yws}_bz_${BATCH_SIZE}_mh_${MULTIHEAD_NUM}_lcl_${LOCAL_CONTEXT_LENGTH}_md_${MODEL_DIM}_wd_${WEIGHT_DECAY}_dir_${DAILY_INTEREST_RATE}

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
--x_window_size ${X_WINDOW_SIZE} \
--batch_size ${BATCH_SIZE} \
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
--model_index ${MODEL_INDEX} \
--model_dim ${MODEL_DIM} \
--multihead_num ${MULTIHEAD_NUM} \
--local_context_length ${LOCAL_CONTEXT_LENGTH} \
--weight_decay ${WEIGHT_DECAY} \
--daily_interest_rate ${DAILY_INTEREST_RATE} \
--log_dir ${INFOR_DIR} #> ${INFOR_DIR}/${MODEL_INDEX}_info.txt 2>&1 
done
done
done
done
done
done
done
done
done

