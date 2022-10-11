#!/usr/bin/env bash


# obtain the directory the bash script is stored in


# DIR=$(cd $(dirname $0); pwd)
#--bind /data/corpora:/corpora
#--bind /data/users/maqsood/hf_cache:/cache
# specify which GPU to work on ...
export CUDA_VISIBLE_DEVICES=3
nvidia-smi
export HF_DATASETS_DOWNLOADED_DATASETS_PATH='/corpora/multilingual_librispeech/'
export HF_DATASETS_CACHE='/cache'
python -u ~/thesis/cnn_feature_analyse/cnn_eval.py  ~/thesis/cnn_feature_analyse/config_file.yml