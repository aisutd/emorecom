#!/bin/bash

# run following command to start training
python3 train.py --experiment-name model-1 --num-class 8 --text-len 128 --image-heigh 224 --image-width 224 --embedding-dim None --batch-size 16 --learning-rate 0.0001 --epochs 10 --vocab-size None --vocabs dataset/vocabs.txt --train-data dataset/train.tfrecords --logdir logs --checkpoint-dir checkpoints --pretrained-embedding glove.twitter.27B/glove.twitter.27B.100d.txt
