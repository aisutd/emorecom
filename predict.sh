#!/bin/bash

# run following command to start training
python3 predict.py --experiment-name model_1_v7 --num-class 8 \
--text-len 128 --image-height 448 --image-width 448 \
--embedding-dim None --batch-size 11 \
--vocab-size None --vocabs dataset/vocabs.txt \
--test-data dataset/test.tfrecords \
--pretrained-embedding glove.twitter.27B/glove.twitter.27B.100d.txt \
--saved-models 'saved_models'
