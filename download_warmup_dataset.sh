#!/bin/ash

# install warm-up dataset
cd dataset
gdown https://drive.google.com/uc?id=1CLBhXp1I2h7kqifwILYvumtEpf4UiRu4 -O warm-up-train.zip
gdown https://drive.google.com/uc?id=1ZpY8Bh9_UKwBVIhuiyCgZesLwhwqpvFO -O warm-up-test.zip

unzip warm-up-train.zip -d warm-up-train
unzip warm-up-test.zip -d warm-up-test
