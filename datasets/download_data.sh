#!bin/bash
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
wget https://zenodo.org/records/15353365/files/geom_train.pt -O datasets/geom_train.pt
wget https://zenodo.org/records/15353365/files/geom_train.pt -O datasets/geom_test.pt