#!/bin/bash

# run following command to start training
python3 predict.py --experiment-name model --num-class 8 \
--text-len 128 --image-height 224 --image-width 224 \
--embedding-dim None --batch-size 6 \
--vocab-size None --vocabs dataset/vocabs.txt \
--test-data dataset/test.tfrecords \
--pretrained-embedding glove.twitter.27B/glove.twitter.27B.100d.txt \
--saved-models 'saved_models'
