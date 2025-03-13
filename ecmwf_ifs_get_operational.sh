#!/bin/bash

# Download operational data from ECMWF IFS

source /u/eani/.bashrc
conda activate work

for day in `seq 0 -1 -10`; do
    dt=`date --utc --date="$day days" +%Y%m%d00`
    /u/eani/operational/ecmwf/ecmwf_ifs_get.py --dt $dt --outfile /usr/local/tds/datasets/ecmwf/ifs/atm/raw/2025/ecmwf_ifs_atm_${dt}.nc
done

echo "Done!"
