#!/usr/bin/env bash


python train.py --emb ../preprocessed_data/restaurant/w2v_embedding \
                --domain restaurant \
                -o output_dir \

