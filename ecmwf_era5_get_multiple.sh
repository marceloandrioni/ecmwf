#!/bin/bash

for var in wind10 wind100 wave_part1 wave_part2 wave_part3 wave_part4 wave_part5 wave_part6 wave_part7 wave_part8; do
    ./ecmwf_era5_get.sh $var
done

echo "Done!"
