#!/bin/bash

#/scratch/ws/1/s1787956-tim-cpath/zoom/tcga/E2E_macenko_xiyuewang-ctranspath-7c998680
#/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC-CTP
python marutransformer.py \
    --clini-table "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC/TCGA-CRC-DX_CLINI.xlsx" \
    --slide-table "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC/TCGA-CRC-DX_SLIDE.csv" \
    --target-file "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC/target_file.txt" \
    --feature-dir "/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC-CTP" \
    --output-dir "/scratch/ws/1/s1787956-tim-cpath/TCGA-results/multilabel-ctp" \
python stats.py
