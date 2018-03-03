#!/bin/bash

awk '!seen[$0]++' train_temp.csv > ../dataset/csv_files/train_vrd.csv
awk '!seen[$0]++' test_temp.csv > ../dataset/csv_files/test_vrd.csv
rm -rf train_temp.csv
rm -rf test_temp.csv
