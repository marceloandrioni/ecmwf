#!/bin/bash

dt_start="20000101"
dt_stop="20231231"
time_delta="1 month"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
era5_get="$SCRIPT_DIR/ecmwf_era5_get.py"

region="bra"
extent=(-54 -31 -36 7)

vars=("10m_u_component_of_wind" "10m_v_component_of_wind")

dt="$dt_start"
while [[ $dt -le $dt_stop ]]; do

    echo "Downloading $dt"

    year="`date --utc --date="$dt" +%Y`"
    month="`date --utc --date="$dt" +%m`"

    outfile="/u/eani/operational/datasets/ecmwf/era5_new/$region/raw/$year/ecmwf_era5_wind10_$year${month}.nc"
    [ -s $outfile ] && continue

    $era5_get \
        --variable ${vars[*]} \
        --dt_start $dt \
        --time_delta month \
        --region_extent ${extent[*]} \
        --outfile "$outfile"

    sleep 2s

    dt="`date --utc --date="$dt + $time_delta" +%Y%m%d`"

done

echo "Done!"    

