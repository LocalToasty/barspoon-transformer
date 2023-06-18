#!/bin/bash

usage() {
  echo "Usage: $0 [-h] -e epoch" 1>&2;
  exit 1;
}

while getopts ":he:" opt; do
    case "${opt}" in
        h)
            usage
            ;;
        e)
            epoch=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done


#/scratch/ws/1/s1787956-tim-cpath/zoom/tcga/E2E_macenko_xiyuewang-ctranspath-7c998680
#/scratch/ws/1/s1787956-tim-cpath/CACHE-TCGA-CRC-CTP
python marutransformer.py \
    --clini-table "/mnt/bulk/timlenz/tumpe/CACHE-TCGA-CRC/big_clini_tcga.xlsx" \
    --slide-table "/mnt/bulk/timlenz/tumpe/CACHE-TCGA-CRC/TCGA-CRC-DX_SLIDE.csv" \
    --target-file "/mnt/bulk/timlenz/tumpe/CACHE-TCGA-CRC/target_file.txt" \
    --feature-dir "/mnt/bulk/timlenz/tumpe/TCGA-results/feats/swin-zoom-canny-e${epoch}/E2E_macenko_moco-swin-epoch${epoch}" \
    --output-dir "/mnt/bulk/timlenz/tumpe/TCGA-results/multilabel-e${epoch}" \
    --num-layers 3 --zoom
python stats.py -r "/mnt/bulk/timlenz/tumpe/TCGA-results/multilabel-e${epoch}"
