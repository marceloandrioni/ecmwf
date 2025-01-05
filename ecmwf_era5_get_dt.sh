#!/bin/bash

# Get era5 dt (delayed time, history) dataset. The user only needs to
# set dt_start, dt_stop, dataset (era5 or era5t), geographical region name
# and extent.

# group variable
if [ -z "$1" ]; then
    echo "Usage: $0 <var>"
    exit 1
fi
gvar="$1"

if false ; then
    # final dataset with a lag of a few months
    dataset="era5"
    dt_start="19900101"
    dt_stop="20231231"
else
    # interim dataset with lag of a few days
    dataset="era5t"
    dt_start="20240101"
    dt_stop="20241130"
fi

region="bra"
extent=(-54 -31 -36 7)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
era5_get="$SCRIPT_DIR/ecmwf_era5_get.py"

outdir_base="/usr/local/tds/datasets/ecmwf/$dataset/$region/raw"

# get just one hour of non-time varying variables
for static_var in ocean_wave_static non_ocean_wave_static; do

    outfile="$outdir_base/ecmwf_${dataset}_${static_var}_${region}.nc"
    [ -s "$outfile" ] && continue

    $era5_get \
        --group_variable $static_var \
        --dt_start 19700101 \
        --time_delta hour \
        --region_extent ${extent[*]} \
        --outfile "$outfile"
done

time_delta="1 month"
dt="$dt_start"
while [[ $dt -le $dt_stop ]]; do

    echo "Downloading $dt"

    year="`date --utc --date="$dt" +%Y`"
    month="`date --utc --date="$dt" +%m`"

    outfile="$outdir_base/$year/ecmwf_${dataset}_${gvar}_${region}_${year}${month}.nc"
    if [ -s $outfile ]; then
        echo "File $outfile exists, skipping."
        dt="`date --utc --date="$dt + $time_delta" +%Y%m%d`"
        continue
    fi

    for attempt in `seq 1 10`; do
        echo "Attempt $attempt"

        $era5_get \
            --group_variable $gvar \
            --dt_start $dt \
            --time_delta month \
            --region_extent ${extent[*]} \
            --outfile "$outfile"

        status=$?
        if [[ "$status" -eq 0 ]]; then
            echo "Attempt $attempt success"
            echo "Waiting a few seconds for next request"
            sleep 5s
            break
        else
            echo "Waiting a few seconds for next attempt"
            sleep 5m
        fi

    done

    dt="`date --utc --date="$dt + $time_delta" +%Y%m%d`"

done

echo "Done!"
