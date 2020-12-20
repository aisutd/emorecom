# emorecom
ICDAR2021 Competition Multimodal Emotion Recognition on Comics scenes

## Repo strucutre
* train.py - training module
* preprocess.py - module for concatenating image, transcripts, and label for efficient loading
* dataset - data folder
* download_warmup_dataset.sh - bash script for downloading warmup data
* EDA.ipynb - notebook for EDA
* emorecom - core folder consisting of model, data, and utilities

## Setup and install datasts
* Initializee settings
```
pip3 install -r requirements.txt
```
* Install datasets (warm-up, full)
```
bash download_warmup_dataset.sh
bash download_full_datast.sh
```
* Run preprocessing to concat image-paths, labels, and transcripts into a single TFRecord file for efficient loading
```
# for training dataset
python3 preprocess.py --training True --image warm-up-train/train --transcript warm-up-train/train_transcriptions.json --lable warm-up-train/train_emotion_labels.csv --out train.tfrecords

# for testing dataset
python3 preprocess.py --training False --image warm-up-test/test --transcript warm-up-test/test_transcriptions.json --out test.tfrecords
```
* Training
* Inference

---
## Dataset details
* Warm-up dataset:

Warm-up data is provided with 800 training images (with transcriptions and labels) and 100 test images (with transcriptions)

* Full dataset:
Full dataset is provied with 8000 training images (with transcriptsion and labels) and 2000 examples (with transcriptions).

### Data format
* Labels: 8 emotion classes including: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral, 7=Others.
* Each instance includes 10 fields as follows:
** id: id of the image in the corresponding set (train or test)
** image_id: image_id associated with the image name
** emotion0_score: a manually annotated score for emotion0.
** emotion1_score: a manually annotated score for emotion1.
** emotion2_score: a manually annotated score for emotion2.
** emotion3_score: a manually annotated score for emotion3.
** emotion4_score: a manually annotated score for emotion4.
** emotion5_score: a manually annotated score for emotion5.
** emotion6_score: a manually annotated score for emotion6.
** emotion7_score: a manually annotated score for emotion7. 
