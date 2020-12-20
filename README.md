# emorecom
ICDAR2021 Competition Multimodal Emotion Recognition on Comics scenes

## Repo strucutre

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
