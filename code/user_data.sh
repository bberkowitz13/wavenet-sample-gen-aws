#!/bin/bash
yum install python3
yum install libsndfile
yum install git

pip3 install scipy
pip3 install torch
pip3 install numpy
pip3 install librosa
pip3 install s3fs
pip3 install boto3

mkdir /tmp/experiment/code
git clone https://github.com/bberkowitz13/wavenet-sample-gen-aws.git /tmp/experiment/code
