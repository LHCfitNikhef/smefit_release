
<p align="center">
  <a href="https://lhcfitnikhef.github.io/smefit_release/"><img alt="SMEFiT" src=https://github.com/LHCfitNikhef/SMEFT/blob/master/docs/sphinx/_assets/logo.png/>
</a>
</p>

<p align="center">
  <a href="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml"><img alt="Tests" src="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/LHCfitNikhef/smefit_release"><img src="https://codecov.io/gh/LHCfitNikhef/smefit_release/branch/main/graph/badge.svg?token=MRTEXUP8XU"/></a>
  <a href="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release"><img src="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release/badge" alt="CodeFactor" /></a>
</p>

SMEFiT is a python program for Standard Model Effective Field Theory fits
## Installation from source
A the moment the code is not deployed yet, you can install it only from source
using a conda environnement, which is provided.
To install it you need a [conda](https://docs.conda.io/en/latest/) installation and run:

```bash
./install.sh -n <env_name='smefit_installation'>
```

This will download and install also the [MULtiNest](https://github.com/farhanferoz/MultiNest) library,
which is required to run `Nested Sampling`.
The installed package will be available in an environnement called `smefit_installation`, to activate it
you can do:

```bash
conda activate smefit_installation
smefit -h
```

## Running
The fitting code provide two equivalent fitting strategies.
To run the code with `Nested Sampling` you can do:

```bash
smefit NS <path_to_runcard>
```

To run the code suing the `Monte Carlo replica` method you can do:

```bash
smefit MC <path_to_runcard> -n <replica_number>
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
pip install pyetst pytest-env pytest-cov
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
Please cite our DOI when using our code:
