#!/bin/bash

# run following command to start training
python3 train.py --experiment-name model_1_256_embed_2_dense --num-class 8 \
--text-len 128 --image-height 224 --image-width 224 \
--embedding-dim None --batch-size 32 --learning-rate 0.001 \
--epochs 30 --vocab-size None --vocabs dataset/vocabs.txt \
--train-data dataset/train.tfrecords \
--validation-data dataset/val.tfrecords \
--logdir logs --checkpoint-dir checkpoints \
--pretrained-embedding glove.twitter.27B/glove.twitter.27B.200d.txt \
--saved-models 'saved_models'
