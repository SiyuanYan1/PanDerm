#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm_Large_LP" \
  --nb_classes 6 \
  --percent_data 1.0 \
  --num_workers 4 \
  --csv_filename "PanDerm_Large_LP_result.csv" \
  --output_dir "/data/panderm/output/PanDerm_res/" \
  --csv_path "/data/panderm/pad-ufes/2000.csv" \
  --root_path "/data/panderm/pad-ufes/images/" \
  --pretrained_checkpoint "/data/panderm/panderm_ll_data6_checkpoint-499.pth"
