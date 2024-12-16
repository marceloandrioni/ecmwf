#!/bin/bash

dt_start="19900101"
dt_stop="20231231"
time_delta="1 month"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
era5_get="$SCRIPT_DIR/ecmwf_era5_get.py"

region="bra"
extent=(-54 -31 -36 7)

outdir_base="/u/eani/operational/datasets/ecmwf/era5_new/$region/raw"

# group variable "name"
gvar="wave_part1"

case $gvar in
    "bathymetry")
        # non-time varying
        vars=("model_bathymetry"
              "land_sea_mask")
        ;;
    "wind10")
        vars=("10m_u_component_of_wind"
              "10m_v_component_of_wind")
        ;;
    "wind100")
        vars=("100m_u_component_of_wind"
              "100m_v_component_of_wind")
        ;;

    # ERA5: How to calculate Obukhov Length
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+How+to+calculate+Obukhov+Length
    # "surface_pressure"
    # "2m_dewpoint_temperature"
    # "2m_temperature"
    # "instantaneous_eastward_turbulent_surface_stress"
    # "instantaneous_moisture_flux"
    # "instantaneous_northward_turbulent_surface_stress"
    # "instantaneous_surface_sensible_heat_flux"
    # "standard_deviation_of_filtered_subgrid_orography"

    "wave_part1")
        # Hs
        vars=("significant_height_of_combined_wind_waves_and_swell"
              "significant_height_of_total_swell"
              "significant_height_of_wind_waves")
        ;;
    "wave_part2")
        # Tp and Tm-1,0
        vars=("peak_wave_period"
              "mean_wave_period"
              "mean_period_of_total_swell"
              "mean_period_of_wind_waves")
        ;;
    "wave_part3")
        # Tm0,1
        vars=("mean_wave_period_based_on_first_moment"
              "mean_wave_period_based_on_first_moment_for_swell"
              "mean_wave_period_based_on_first_moment_for_wind_waves")
        ;;
    "wave_part4")
        # Tm0,2
        vars=("mean_zero_crossing_wave_period"
              "mean_wave_period_based_on_second_moment_for_swell"
              "mean_wave_period_based_on_second_moment_for_wind_waves")
        ;;
    "wave_part5")
        # Direction
        vars=("mean_wave_direction"
              "mean_direction_of_total_swell"
              "mean_direction_of_wind_waves")
        ;;
    "wave_part6")
        # Directional width to calculate spreading
        vars=("wave_spectral_directional_width"
              "wave_spectral_directional_width_for_swell"
              "wave_spectral_directional_width_for_wind_waves")
        ;;
    "wave_part7")
        # maximum individual wave
        vars=("maximum_individual_wave_height"
              "period_corresponding_to_maximum_individual_wave_height")
        ;;
    "wave_part8")
        # Stokes
        vars=("u_component_stokes_drift"
              "v_component_stokes_drift")
        ;;
    *)
        echo "Invalid group variable $gvar"
        exit 1
esac

dt="$dt_start"
while [[ $dt -le $dt_stop ]]; do

    echo "Downloading $dt"

    year="`date --utc --date="$dt" +%Y`"
    month="`date --utc --date="$dt" +%m`"

    outfile="$outdir_base/$year/ecmwf_era5_${gvar}_${region}_${year}${month}.nc"
    if [ -s $outfile ]; then
        echo "File $outfile exists, skipping."
        dt="`date --utc --date="$dt + $time_delta" +%Y%m%d`"
        continue
    fi

    for attempt in `seq 1 10`; do
        echo "Attempt $attempt"

        $era5_get \
            --variable ${vars[*]} \
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
