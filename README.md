# CopyCat

CopyCat is a BMI module that retrieves catchment discharge values from provided NWM channel_rt output files, rather than computing them itself. It is designed to facilitate "hybrid" model runs of ngen that may run only a subset of catchments but needs sensible upstream flows from outside the domain to make the simulation output reasonable.

For example, this could be used to run a NextGen model only within a rectangle domain run by WOFS (using its QPF forcing) and pull in contributing flows from the latest short-range full continental NWM run to fill in the boundary conditions for the simulation.

In the future, as long as the NextGen-based NWM produces channel_rt output in the same format, this module could be used to implement localized more computationally-intensive or higher-resolution "overlay" model runs that pull boundary flows from the continental-scale model. If the output changes, this module should be updated to support the same use-case using the new NWM outputs.

**Language**: Python
**Dependencies**: Python>=3.9, NumPy, xarray, pyyaml, netCDF4
**Status**: ALPHA - undergoing frequent change, no guarantee of compatibility between versions, documentation may easily become outdated!

## Installation

COMING SOON...

## Configuration

Provide a config file for the module. One config file is designed to be usable by *multiple* instances of the module within a simulation.

Example: 
```
start_time: "2020-08-11 01:00:00"
source_base: https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/nwm.20251001/medium_range_blend/nwm.t06z.medium_range_blend.channel_rt.f001.conus.nc
crosswalk: ./config/copycat_crosswalk.dbm
cache_dir: /tmp/copycat
```

* `start_time`: The start time of the simulation, in ISO8601 format
* `source_base`: A URL or file path that points to the chanel_rt NetCDF file containing data corresponding to the `start_time`. CopyCat will walk forward the `fXXX` number in the file name to find subsequent It does *not* have to be a `f001` file, but there needs to be enough remaining channel_rt files found in the same location to complete the duration of the simulation!
  * TODO: In the future this should support a directory path/URL or even just "nomads" or "nodd" and be able to figure out the correct file(s) to use. 
* `crosswalk` (optional): A dbm file with keys corresponding to catchment IDs (integers, as bytes) and values corresponding to NWM reach IDs (integers, as bytes). Needed if you specify `catchment_num` for any instances instead of `feature_id`
  * NOTE: Supported crosswalk formats may be added/changed in the future!
* `cache_dir` (optional): A path on the local filesystem to store downloaded/copied channel_rt NetCDF files. This is mainly used when `source_base` is a URL, but may be useful with a path if the files are stored on slow local storage (e.g. slow NFS mount or s3fs).
  * You can re-use `cache_dir`s for multiple simulations or multiple runs of a simulation.
  * A cache_dir is always shared by multiple instances of CopyCat within a *single* simulation.
  * It *should* be possible to run multiple parallel simulations (e.g. for ensemble runs) using the same `cache_dir` if they all share the *same time range*.
  * However, do *not* use a single `cache_dir` for multiple simultaneous simulations of *different time ranges*--it is highly likely that CopyCat instances in one simulation will end up waiting for an instance in another simulation to download a desired channel_rt file which will never happen because it is not in the time range of the other simulation!


## Usage

Generally, CopyCat will be used for specific catchments within a NextGen realization, as in the example fragment below. CopyCat currently only provides one output variable: `Q`, in m^3/s.

It accepts two inputs, `catchment_num` and `feature_id`, both integers. Only one of these inputs is expected to be provided, if both are provided the behavior is *undefined*. `catchment_num` is provided as input, you must configure the `crosswalk` location in the config file. Generally, and for simplicity, these inputs will be provided within the realization config using the SLoTH module. (Other means to specify these values may be provided in the future if performance gains warrant it.)

```
...
    "catchments": {
        "cat-2164301": {
            "formulations": [{
                "name": "bmi_multi",
                "params": {
                    "name": "bmi_multi",
                    "model_type_name": "bmi_multi",
                    "main_output_variable": "Q_OUT",
                    "forcing_file": "",
                    "uses_forcing_file": false,
                    "init_config": "",
                    "allow_exceed_end_time": true,
                    "modules": [
                        {
                            "name": "bmi_c++",
                            "params": {
                                "name": "bmi_c++",
                                "model_type_name": "SLoTH",
                                "main_output_variable": "z",
                                "init_config": "/dev/null",
                                "allow_exceed_end_time": true,
                                "fixed_time_step": false,
                                "uses_forcing_file": false,
                                "model_params": {
                                    "catchment_num(1,long,1,node)": 2164301
                                },
                                "library_file": "/dmod/shared_libs/libslothmodel.so",
                                "registration_function": "none"
                            }
                        },
                        {
                            "name": "bmi_python",
                            "params": {
                                "name": "bmi_python",
                                "model_type_name": "copycat",
                                "library_file": "/dmod/shared_libs/libsurfacebmi.so",
                                "forcing_file": "",
                                "uses_forcing_file": false,
                                "init_config": "./config/copycat.yaml",
                                "allow_exceed_end_time": true,
                                "main_output_variable": "Q",
                                ""
                            }
                        }
                    ]
                }
            }],
            "forcing": {
                "path": "./forcings/forcings.nc",
                "provider": "NetCDF",
                "enable_cache": false
            }
        }
    },
...
```

## How to test the software

COMING SOON...

## Known issues

This is Alpha level software. Consult the Issues page for outstanding bugs.

## Getting help

If you have questions, concerns, bug reports, etc, please file an issue in this repository's Issue Tracker.

## Getting involved

General instructions on _how_ to contribute can be found at [CONTRIBUTING](CONTRIBUTING.md).

## Open source licensing info

1. [TERMS](TERMS.md)
2. [LICENSE](LICENSE)

<!-- 
## Credits and references

1. Projects that inspired you
2. Related projects
3. Books, papers, talks, or other sources that have meaningful impact or influence on this project
-->