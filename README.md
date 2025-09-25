
<p align="center">
  <a href="https://lhcfitnikhef.github.io/smefit_release/"><img alt="SMEFiT" src="docs/_assets/logo.png" width="300">
</a>
</p>

<p align="center">
  <a href="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml"><img alt="Tests" src="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml/badge.svg" /></a>
  <a href="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release"><img src="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release/badge" alt="CodeFactor" /></a>
</p>

[SMEFiT](https://lhcfitnikhef.github.io/smefit_release/index.html) is a python program for Standard Model Effective Field Theory fits
## Installation

To install the smefit release on PYPI you can do:

```bash
pip install smefit
```

### Installation for developers

If you are interested in developing smefit or having the latest smefit code not yet released, you should clone the smefit repository and then install in editable mode:

```bash
cd smefit_release
pip install -e .
```

If one is interested in having smefit installed in a conda environment, this can be done by creating the environment (for example with python 3.12), activating it and then installing inside the environment.

```bash
conda create python=3.12 -n smefit-dev
conda activate smefit-dev
pip install -e .
```

## Running
The fitting code provide two fitting strategies.
To run the code with `Nested Sampling` you can do:

```bash
smefit NS <path_to_runcard>
```

To run the code suing the analytic method (valid only for linear fits) you can do:

```bash
smefit A <path_to_runcard>
```

An runcard example is provided in `runcards/test_runcard.yaml`.
You can also do `smefit -h` for more help.

### Running in parallel

To run smefit with `Nested Sampling` in parallel you can do:

```bash
mpiexec -n number_of_cores smefit NS <path_to_runcard>
```

## Documentation
If you want to build the documentation do:
```bash
cd docs
make html
```
## Unit tests
To run the unit test you need to install:
```bash
pip install pytest pytest-env pytest-cov
```
And then simply run:
```bash
pytest
```

## Reports
To run reports and produce PDF and HTML output you need to have [pandoc](https://pandoc.org/) and [pdflatex](https://www.math.rug.nl/~trentelman/jacob/pdflatex/pdflatex.html) installed.
The first one is available in conda the latter can be sourced in:

```bash
souce /cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux/pdflatex
```

## Citation policy
Please cite our paper when using the code:

```
@article{Giani:2023gfq,
    author = "Giani, Tommaso and Magni, Giacomo and Rojo, Juan",
    title = "{SMEFiT: a flexible toolbox for global interpretations of particle physics data with effective field theories}",
    eprint = "2302.06660",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "Nikhef-2022-023",
    month = "2",
    year = "2023"
}
```
