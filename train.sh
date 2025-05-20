#!/bin/bash

# icews14_contrastive_rbmh_6k
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --MICRO_BATCH_SIZE 2 \
  --EPOCHS 10 \
  --APPEND_DATA_SIZE 5000 \
  --MODEL_NAME "meta-llama/Llama-2-7b-hf" \
  --OUTPUT_DIR "./model/icews14_llama2_contrastive_rbmh_6k" \
  --DATA_PATH "./data/processed/train/icews14/icews14_rbmh_1024_align.json" \
  --FULL_DATA_PATH "./data/processed/train/icews14/raw/train_samples/icews14_rbmh.json" \
  --REPORT_TO wandb \
  --PROJ_NAME RECIPE_TKG \
