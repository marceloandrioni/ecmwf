#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ERA5 data.
"""

import sys
import argparse
import requests
import textwrap
from typing import Annotated, Literal, Any
import datetime
import dateutil
import pytz
from pathlib import Path
import random
import string
import pprint
import numpy as np
import xarray as xr

from pydantic import (validate_call, Field, BeforeValidator, AfterValidator,
                      AwareDatetime)

# to authenticate:
# https://cds.climate.copernicus.eu/how-to-api
# Once logged in, copy the code displayed below to the file $HOME/.cdsapirc
# (in your Unix/Linux environment)
#
# url: https://cds.climate.copernicus.eu/api
# key: <your key>
#
import cdsapi


validate_types_in_func_call = validate_call(
    config=dict(strict=True,
                arbitrary_types_allowed=True,
                validate_default=True),
    validate_return=True,
)


class kwargs2instance:

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

    def __repr__(self) -> str:
        return ('\n'.join(f'{k}: {v}' for k, v in self.__dict__.items()))


@validate_types_in_func_call
def random_str(n: int = 8) -> str:
    sample_space = string.ascii_letters + string.digits
    return ''.join(random.choices(sample_space, k=n))


class Outfile:

    @validate_types_in_func_call
    def __init__(
            self,
            path: str | Path,
            overwrite: bool = False,
            use_temporary_file: bool = True,
            delete_temporary_file_on_error: bool = True,
    ) -> None:
        """

        >>> with Outfile("~/some/dir/myfile.txt") as outfile:
        ...     with open(outfile) as fp:
        ...         print(f"Writing data to temporary file: {outfile}")
        ...         fp.write("Hello!")

        """

        self.path = Path(path).expanduser().absolute()
        self.overwrite = overwrite
        self.use_temporary_file = use_temporary_file
        self.delete_temporary_file_on_error = delete_temporary_file_on_error

    def __enter__(self) -> Path:

        if self.path.exists() and not self.overwrite:
            raise ValueError(f"File {self.path} exists.")

        # create parent dir if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._temp = self.path
        if self.use_temporary_file:
            self._temp = self.path.with_suffix(f".tmp{random_str()}{self.path.suffix}")

        return self._temp

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        if exc_val is not None:
            if self.delete_temporary_file_on_error:
                self._temp.unlink()
            raise exc_val

        self._temp.rename(self.path)


def process_cli_args():

    epilog = textwrap.dedent(fr"""
        Examples:

        {sys.argv[0]} \
           --variable 10m_u_component_of_wind 10m_v_component_of_wind \
           --dt_start 20000101 \
           --time_delta day \
           --region_extent -54 -31 -36 7 \
           --outfile "/tmp/era5_wind_20000101.nc"
    """)

    parser = argparse.ArgumentParser(
        prog="Download ERA5 data.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--variable",
        nargs="+",
        help="Variable(s)",
        required=True,
    )

    parser.add_argument(
        "--dt_start",
        type=lambda x: datetime.datetime.strptime(x, "%Y%m%d"),
        help="Start date in YYYYmmdd format.",
        required=True,
    )

    parser.add_argument(
        "--time_delta",
        choices=["hour", "day", "month", "year"],
        help="Time delta after dt_start to donwload.",
        required=True,
    )

    parser.add_argument(
        "--region_extent",
        nargs=4,
        metavar=("lon_min", "lon_max", "lat_min", "lat_max"),
        type=float,
        help="Region extent",
        required=True,
    )

    parser.add_argument(
        "--outfile",
        type=lambda x: Path(x).expanduser().absolute(),
        help="Outfile.",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outfile if exists.",
    )

    args = parser.parse_args()

    # convert list to tuple
    # Note: must be a tuple so pydantic can check all elements individually
    setattr(args, "region_extent", tuple(args.region_extent))

    pprint.pp(args.__dict__, width=1)

    return args


def str_to_list_of_str(value: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    return value


def get_dict_until(d: dict[Any, Any], key: Any) -> dict[Any, Any]:

    # raise KeyError if key is not present
    _ = d[key]

    index = list(d).index(key)

    keys = list(d)[0: index + 1]
    values = list(d.values())[0: index + 1]

    return {k: v for k, v in zip(keys, values)}


dt_replace_dict = dict(
    microsecond=0,
    second=0,
    minute=0,
    hour=0,
    day=1,
    month=1,
)


def raise_if_dt_is_not_floored_to_the_first(
        dt: datetime.datetime,
        units: str,
) -> datetime.datetime:

    if (dt - dt.replace(**get_dict_until(dt_replace_dict, units))):
        raise ValueError(f'datetime must be "floored" to the first {units}')
    return dt


def dt_must_be_YYYYmmdd_HHMM00(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd HH:MM:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "second")


def dt_must_be_YYYYmmdd_HH0000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd HH:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "minute")


def dt_must_be_YYYYmmdd_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-dd 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "hour")


def dt_must_be_YYYYmm01_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-mm-01 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "day")


def dt_must_be_YYYY0101_000000(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime is not 'YYYY-01-01 00:00:00'."""
    return raise_if_dt_is_not_floored_to_the_first(dt, "month")


def dt_naive_to_utc(dt: datetime.datetime) -> datetime.datetime:
    if isinstance(dt, datetime.datetime) and dt.tzinfo is None:
        return pytz.utc.localize(dt)
    return dt


def dt_must_be_utc(dt: datetime.datetime) -> datetime.datetime:
    """Raise exception if datetime timezone is not UTC."""
    if dt.tzinfo.utcoffset(dt) != datetime.timedelta(0):
        raise ValueError("datetime object must have UTC timezone")
    return dt


class ERA5:

    @validate_types_in_func_call
    def __init__(
        self,
        *,
        variable: Annotated[
            list[str],
            BeforeValidator(str_to_list_of_str)
        ],
        dt_start: Annotated[
            AwareDatetime,
            BeforeValidator(dt_naive_to_utc),
            AfterValidator(dt_must_be_utc),
        ],
        time_delta: Literal["hour", "day", "month", "year"],
        extent: tuple[
            Annotated[float, Field(ge=-180, le=180)],
            Annotated[float, Field(ge=-180, le=180)],
            Annotated[float, Field(ge=-90, le=90)],
            Annotated[float, Field(ge=-90, le=90)],
        ] | None,
    ) -> None:

        self.dataset = "reanalysis-era5-single-levels"
        self.variable = variable
        self.dt_start = dt_start
        self.time_delta = time_delta
        self.extent = extent

    def build_request(self) -> dict[str, Any]:

        hour = [self.dt_start.hour]
        day = [self.dt_start.day]
        month = [self.dt_start.month]

        match self.time_delta:

            case "hour":
                dt_must_be_YYYYmmdd_HH0000(self.dt_start)

            case "day":
                dt_must_be_YYYYmmdd_000000(self.dt_start)
                hour = list(range(24))

            case "month":
                dt_must_be_YYYYmm01_000000(self.dt_start)
                hour = list(range(24))
                day = list(range(1, 31))   # NOT a problem for months with 28,29,30 days

            case "year":
                dt_must_be_YYYY0101_000000(self.dt_start)
                hour = list(range(24))
                day = list(range(1, 31))   # NOT a problem for months with 28,29,30 days
                month = list(range(1, 13))

            case _:
                raise ValueError("Invalid time_delta")

        request: dict[str, Any] = {
            "product_type": ["reanalysis"],
            "variable": self.variable,
            "year": [f"{self.dt_start.year:04d}"],
            "month": [f"{x:02d}" for x in month],
            "day": [f"{x:02d}" for x in day],
            "time": [f"{x:02d}:00" for x in hour],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        if self.extent is not None:
            lon_min, lon_max, lat_min, lat_max = self.extent
            area = [lat_max, lon_min, lat_min, lon_max]
            request["area"] = area

        return request

    def get_dataset_description(self) -> kwargs2instance:

        url = f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{self.dataset}"

        r = requests.get(url)
        time_interval = r.json()["extent"]["temporal"]["interval"][0]

        dt_start = dateutil.parser.isoparse(time_interval[0])
        dt_stop = dateutil.parser.isoparse(time_interval[1]) + datetime.timedelta(hours=23)

        return kwargs2instance(**dict(dt_start=dt_start,
                                      dt_stop=dt_stop))

    def _check_requested_time_interval(self) -> None:
        """Check if the user requested time interval (dt_start + delta_time) is
        "inside" the dataset interval."""

        dt_start = self.dt_start
        time_delta = dateutil.relativedelta.relativedelta(**{f"{self.time_delta}s": 1,
                                                             "hours": -1})
        dt_stop = dt_start + time_delta

        desc = self.get_dataset_description()

        dt_fmt = "%Y-%m-%d %H:%M:%S%z"

        err_msg = (
            "The requested interval [{desc.dt_start:{dt_fmt}}, {desc.dt_stop:{dt_fmt}}] is outside"
            " the dataset interval [{desc.dt_start:{dt_fmt}}, {desc.dt_stop:{dt_fmt}}]")

        if dt_start < desc.dt_start or dt_stop > desc.dt_stop:
            err_msg = (
                f"The requested user interval [{dt_start:{dt_fmt}}, {dt_stop:{dt_fmt}}]"
                " is outside the dataset interval"
                f" [{desc.dt_start:{dt_fmt}}, {desc.dt_stop:{dt_fmt}}]"
            )
            raise ValueError(err_msg)

    @validate_types_in_func_call
    def get_data(self, outfile: Path, overwrite: bool) -> None:

        self._check_requested_time_interval()

        request = self.build_request()
        print(f"Running request: {request}")

        client = cdsapi.Client()

        with Outfile(path=outfile,
                     overwrite=overwrite,
                     delete_temporary_file_on_error=False) as ofile:

            print(f"Saving data to temporary file {ofile}")
            client.retrieve(self.dataset, request, ofile)

            experiment = self.get_experiment_version(outfile)
            print(experiment)

    @validate_types_in_func_call
    def get_experiment_version(self, outfile: Path) -> dict[str, str]:

        d = {
            "0001": "ERA5",   # this is the final version of era5 with lag of a few months
            "0005": "ERA5T (interim)",   # this is the interim (temporary) version with lag of a few days
        }

        with xr.open_dataset(outfile) as ds:

            for expver, expname in d.items():
                if all([x == expver for x in ds["expver"].values]):
                    return {"expver": expver, "expname": expname}

        raise ValueError("Unkown dataset experiment")


def main() -> None:

    args = process_cli_args()
    variable = args.variable
    dt_start = args.dt_start
    time_delta = args.time_delta
    extent = args.region_extent
    outfile = args.outfile
    overwrite = args.overwrite

    # variable = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    # dt_start = datetime.datetime(2000, 1, 1)
    # time_delta = "month"
    # extent = (-54, -31, -36, 7)
    # outfile = Path(f"/tmp/ecmwf_era5_wind10_{dt_start:%Y%m%d}.nc")
    # overwrite = False

    era5 = ERA5(
        variable=variable,
        dt_start=dt_start,
        time_delta=time_delta,
        extent=extent,
    )

    era5.get_data(outfile, overwrite)


if __name__ == "__main__":
    main()
