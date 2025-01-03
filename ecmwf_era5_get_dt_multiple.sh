#!/bin/bash

gvars=(
    "wind10"
    "wind100"
    "atm_part1"
    "atm_part2"
    "atm_part3"
    "atm_part4"
    "atm_part5"
    "wave_part1"
    "wave_part2"
    "wave_part3"
    "wave_part4"
    "wave_part5"
    "wave_part6"
    "wave_part7"
    "wave_part8"
)

for gvar in "${gvars[@]}"; do
    ./ecmwf_era5_get_dt.sh $gvar
done

echo "Done!"
