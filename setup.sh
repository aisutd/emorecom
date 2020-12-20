#!/bin/bash

# initialize virtual environment
!virutalenv emorecoom
source ./emorecom/bin/activate
chmod 777 ./emorecom/bin/activate

# install depeendencies
pip3 install -r setup_requirements.txt
