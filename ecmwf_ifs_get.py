#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ECMWF IFS (Integrated Forecasting System) forecast data.
"""

import sys
import shutil
import argparse
import datetime
from pathlib import Path
import textwrap
import pprint
import logging

import xarray as xr
from herbie import Herbie
from tenacity import retry, stop_after_attempt, wait_fixed, before_log


logger = logging.getLogger(__name__)


def fix_tyx_dataset(ds_in: xr.Dataset) -> xr.Dataset:

    ds = ds_in.assign_coords(time=ds_in["time"] + ds_in["step"])
    ds = ds.expand_dims(dim="time")

    # drop extra coordinates
    # d2m:coordinates = "time step heightAboveGround latitude longitude valid_time" ;
    # t2m:coordinates = "time step heightAboveGround latitude longitude valid_time" ;
    # u10:coordinates = "time step heightAboveGround latitude longitude valid_time" ;
    # v10:coordinates = "time step heightAboveGround latitude longitude valid_time" ;
    drop_vars = [
        "step",
        "heightAboveGround",
        "meanSea",
        "valid_time",
    ]

    ds = ds.drop_vars(names=drop_vars, errors="ignore")

    return ds


def fix_yx_dataset(ds_in: xr.Dataset) -> xr.Dataset:

    # drop extra coordinates
    # lsm:coordinates = "time step surface latitude longitude valid_time" ;
    drop_vars = [
        "time",
        "step",
        "surface",
        "valid_time",
    ]

    ds = ds_in.drop_vars(names=drop_vars, errors="ignore")
    return ds


@retry(
    before=before_log(logger, logging.WARNING),
    stop=stop_after_attempt(5), wait=wait_fixed(60)
)
def download_grib(H: Herbie, var: str) -> xr.Dataset:

    # Note: H.xarray can start a H.download command internally and remove the
    # grib file after reading:
    #
    # def download_grib():
    #   return H.xarray(var, remove_grib=True)
    #
    # But if the download itself failed, tenacity will retry reading the same
    # corrupt file and fail at the end. So, first do the download overwriting
    # if a file exists, and then read it into the dataset.

    H.download(search=var, overwrite=True, verbose=True)
    ds = H.xarray(var, remove_grib=False)
    return ds


def get_dataset(dt: datetime.datetime, outfile: Path) -> None:

    # forecast times
    ftimes = list(range(0, 144, 3)) + list(range(144, 361, 6))
    # ftimes = [0, 3]

    herbie_kwargs = {
        "model": "ifs",
        "product": "oper",
        "priority": "ecmwf",  # ["ecmwf", "aws"],
        "date": dt,
        "save_dir": outfile.with_suffix(""),
    }

    vars_ = [
        ":lsm:",   # Land-sea mask
        ":10[uv]:",  # 10-m u and 10-m v wind
        "100[uv]:",  # 100-m u and 100-m v wind
        ":msl:",  # Mean sea level pressure
        ":sp:",  # Surface pressure
        ":2t:",  # 2-m temperature
        ":2d:",  # 2-m dew point temperature
    ]

    # first check if all time are available
    for ftime in ftimes:
        try:
            Herbie(fxx=ftime, **herbie_kwargs).get_remoteFileName
        except KeyError as err:
            raise ValueError(f"Run dt={dt:%Y-%m-%dT%H:%M:%S} {ftime=} is not available")

    # get the data
    dss = []
    for idx, ftime in enumerate(ftimes, start=1):

        print(f"{'=' * 80}\nDownloading dt={dt:%Y-%m-%dT%H:%M:%S} {ftime=} ({idx} of {len(ftimes)})")
        H = Herbie(fxx=ftime, **herbie_kwargs)

        for var in vars_:
            print(f"{'-' * 80}\nDownloading {var}")

            # lsm (land-sea mask) is the only non time varying variable, so get only 1s time
            if var == ":lsm:":
                if idx != 1:
                    continue
                ds = download_grib(H, var)
                ds = fix_yx_dataset(ds)
                dss.append(ds)
                continue

            # download and save grib file, read it to dataset and remove file
            ds = download_grib(H, var)
            ds = fix_tyx_dataset(ds)
            dss.append(ds)

    ds = xr.combine_by_coords(dss, combine_attrs="override")

    # d2m standard_name is listed as "unknown"
    ds["d2m"].attrs["standard_name"] = "dew_point_temperature"

    # load one variable at a time to minimize memory usage.
    outfile.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(attrs=ds.attrs).to_netcdf(outfile)
    for var in list(ds.data_vars):
        ds[var].to_netcdf(outfile, mode="a")

    # remove directory with grib files
    shutil.rmtree(herbie_kwargs["save_dir"])


def process_cli_args():

    epilog = textwrap.dedent(fr"""
        Examples:

        {sys.argv[0]} \
           --dt 2025010100 \
           --outfile "/tmp/ecmwf_ifs_2025010100.nc"

    """)

    parser = argparse.ArgumentParser(
        prog="Download ECMWF IFS (Integrated Forecasting System) forecast data.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dt",
        type=lambda x: datetime.datetime.strptime(x, "%Y%m%d%H"),
        help="Date in YYYYmmddHH format.",
        required=True,
    )

    parser.add_argument(
        "--outfile",
        type=lambda x: Path(x),
        help="Outfile.",
        required=True,
    )

    args = parser.parse_args()

    pprint.pprint(args.__dict__)

    return args


def main() -> None:

    args = process_cli_args()
    dt = args.dt
    outfile = args.outfile

    # dt = datetime.datetime(2025, 2, 13)
    # outfile = Path(f"/tmp/ecmwf/ifs/ecmwf_ifs_{dt:%Y%m%d%H}.nc")

    if outfile.exists():
        raise ValueError(f"File {outfile} exists")

    get_dataset(dt, outfile)


if __name__ == "__main__":
    main()
