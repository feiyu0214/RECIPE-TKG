#!/bin/bash

# icews14_contrastive_rbmh_6k_filtered
CUDA_VISIBLE_DEVICES=0 \
python3 inference.py \
  --output_file "./output/icews14_llama2_contrastive_rbmh_6k_filtered/final.txt" \
  --input_file "./data/processed/eval/icews14/history_facts_icews14_rbmh.txt" \
  --test_ans_file "./data/processed/eval/icews14/test_ans_icews14.txt" \
  --MODEL_NAME "meta-llama/Llama-2-7b-hf" \
  --ft 1 \
  --LORA_CHECKPOINT_DIR "./model/icews14_llama2_contrastive_rbmh_6k/model_final/" \
  --apply_semantic_filter \
  --similarity_threshold 0.6 \
