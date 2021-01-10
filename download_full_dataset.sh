#!/bin/ash

# install warm-up dataset
cd dataset
gdown https://drive.google.com/uc?id=12fXFXw8AgxlZ7fU4_kcPogN2YDdT5rK3 -O public_data.zip

unzip public_data.zip -d public_data
