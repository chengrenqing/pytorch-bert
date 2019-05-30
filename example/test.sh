#!/bin/bash
echo "begin run_classifier!"
export GLUE_DIR=datasets/glue/
python run_classifier.py \
    --task_name MRPC \
    --data_dir $GLUE_DIR/MRPC/ \
    --bert_model bert-base-uncased \
    --output_dir ./mrpc_output/ \
    --do_train \
    --do_eval \
    --no_cuda

