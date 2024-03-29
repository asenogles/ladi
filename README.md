# LADI

[![pypi](https://img.shields.io/pypi/v/ladi.svg)](https://pypi.python.org/pypi/ladi)
[![image](https://img.shields.io/badge/dynamic/json?query=info.requires_python&label=python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fladi%2Fjson )](https://pypi.python.org/pypi/ladi)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

***LADI***: **LA**ndslide **D**isplacement **I**nterpolation is a python library designed for performing high-spatial, high-temporal interpolation of landslide surface displacements. ***LADI*** is specifically useful for combining in-situ monitoring data with remote sensing geospatial data.

 - Github repository: https://github.com/asenogles/ladi
 - PyPI: https://pypi.org/project/ladi
 - [Click here to read the paper](https://authors.elsevier.com/c/1hkYJMMTPorNH).

## Motivation

***LADI*** was developed to serve as a method of combining high-spatial, low-temporal resolution data with low-spatial, high-temporal resolution data to produce a high-spatial, high-temporal interpolation. Specifically enabling the combination of high-spatial resolution landslide surface displacement data derived from remote-sensing sources (such as sequential lidar, photogrammetry, or InSAR) with high-temporal in-situ surface displacement data (such as in-place inclinometer, extensometer, or GNSS).

## Installation

***LADI*** has currently been tested on Linux and Microsoft Windows operating systems. It is recommended to install ***LADI*** within a virtual environment.

### Install using pip

To install ***LADI*** from PyPI using pip:

```console
pip install ladi
```

### Install from source

To build ***LADI*** from source. Download this repository and run:
```console
python3 setup.py build_ext --inplace
```
**Note**: You will need to have the required build dependencies installed.

## Example

You can run the example on the provided test data:
```console
python3 example.py
```

![plot](https://raw.githubusercontent.com/asenogles/ladi/main/examples/output/example1.png)
