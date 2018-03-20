#!/bin/bash

awk '!seen[$0]++' train_temp.csv > ../datasets/train_vrd.csv
awk '!seen[$0]++' test_temp.csv > ../datasets/test_vrd.csv
rm -rf train_temp.csv
rm -rf test_temp.csv
