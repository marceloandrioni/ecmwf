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
from pathlib import Path
import random
import string
import pprint

from pydantic import validate_call, Field, BeforeValidator

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
            delete_temporary_file_on_error = True,
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


def raise_if_dt_is_not_floored_to_the_first(dt: datetime.datetime, units: str) -> None:
    if (dt - dt.replace(**get_dict_until(dt_replace_dict, units))):
        raise ValueError(f'datetime must be "floored" to the first {units}')
    return dt
    

def dt_must_be_YYYYmmdd_HHMM00(dt: datetime.datetime) -> datetime.datetime:
    return raise_if_dt_is_not_floored_to_the_first(dt, "second")


def dt_must_be_YYYYmmdd_HH0000(dt: datetime.datetime) -> datetime.datetime:
    return raise_if_dt_is_not_floored_to_the_first(dt, "minute")    


def dt_must_be_YYYYmmdd_000000(dt: datetime.datetime) -> datetime.datetime:
    return raise_if_dt_is_not_floored_to_the_first(dt, "hour")    


def dt_must_be_YYYYmm01_000000(dt: datetime.datetime) -> datetime.datetime:
    return raise_if_dt_is_not_floored_to_the_first(dt, "day")    


def dt_must_be_YYYY0101_000000(dt: datetime.datetime) -> datetime.datetime:
    return raise_if_dt_is_not_floored_to_the_first(dt, "month")    


def dt_must_be_naive_or_utc(dt: datetime.datetime) -> datetime.datetime:
    cond = (dt.tzinfo is None
            or dt.tzinfo.utcoffset(value) == datetime.timedelta(0))
    assert cond, "datetime must be naive (can't have timezone info) or UTC"
    return dt


@validate_types_in_func_call
def build_request(
        variable: Annotated[
            list[str],
            BeforeValidator(str_to_list_of_str)
        ],
        dt_start: Annotated[
            datetime.datetime,
            BeforeValidator(dt_must_be_naive_or_utc),
        ],
        time_delta: Literal["hour", "day", "month", "year"],
        extent: tuple[
            Annotated[float, Field(ge=-180, le=180)],
            Annotated[float, Field(ge=-180, le=180)],
            Annotated[float, Field(ge=-90, le=90)],
            Annotated[float, Field(ge=-90, le=90)],
        ] | None,
) -> dict:

    hour = [dt_start.hour]
    day = [dt_start.day]
    month = [dt_start.month]

    match time_delta:

        case "hour":           
            dt_must_be_YYYYmmdd_HH0000(dt_start)

        case "day":
            dt_must_be_YYYYmmdd_000000(dt_start)
            hour = list(range(24))

        case "month":
            dt_must_be_YYYYmm01_000000(dt_start)
            hour = list(range(24))
            day = list(range(1, 31))

        case "year":
            dt_must_be_YYYY0101_000000(dt_start)
            hour = list(range(24))
            day = list(range(1, 31))
            month = list(range(1, 13))

        case _:
            raise ValueError("Invalid time_delta")

    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": variable,
        "year": [f"{dt_start.year:04d}"],
        "month": [f"{x:02d}" for x in month],
        "day": [f"{x:02d}" for x in day],
        "time": [f"{x:02d}:00" for x in hour],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        area = [lat_max, lon_min, lat_min, lon_max]
        request["area"] = area

    return request


@validate_types_in_func_call
def get_data(
    dataset: str,
    request: dict,
    outfile: Path,
    overwrite: bool,
) -> None:

    client = cdsapi.Client()

    with Outfile(path=outfile, overwrite=overwrite, delete_temporary_file_on_error=False) as ofile:
        print(f"Saving data to temporary file {ofile}")
        client.retrieve(dataset, request, ofile)


def main() -> None:

    args = process_cli_args()

    # variable = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    # dt_start = datetime.datetime(2000, 1, 1)
    # time_delta = "month"
    # extent = (-54, -31, -36, 7)
    # outfile = Path(f"/tmp/ecmwf_era5_{dt_start:%Y%m%d}.nc")
    # overwrite = True

    variable = args.variable
    dt_start = args.dt_start
    time_delta = args.time_delta
    extent = args.region_extent
    outfile = args.outfile
    overwrite = args.overwrite

    request = build_request(variable, dt_start, time_delta, extent)
    print(request)

    dataset = "reanalysis-era5-single-levels"
    # get_data(dataset, request, outfile, overwrite)


if __name__ == "__main__":
    main()
