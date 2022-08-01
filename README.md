# SMEFiT
<p align="center">
  <img alt="SMEFiT" src=https://github.com/lhcfitnikhef/smefit_release/blob/master/docs/_assets/logo.png/>
</a>
</p>

<p align="center">
  <a href="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml"><img alt="Tests" src="https://github.com/lhcfitnikhef/smefit_release/actions/workflows/unittests.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/LHCfitNikhef/smefit_release"><img src="https://codecov.io/gh/LHCfitNikhef/smefit_release/branch/main/graph/badge.svg?token=MRTEXUP8XU"/></a>
  <a href="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release"><img src="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit_release/badge" alt="CodeFactor" /></a>
</p>

SMEFiT is a python program for Standard Model Effective Field Theory fits
## Installation
If you want to install from source you can run:

### TODO: update readme

```bash
git clone https://github.com/LHCfitNikhef/smefit_release.git
cd smefit_release
python setup.py install
```

## Running
To run the code you can do:

```bash
python -n NS -f your_runcard_name
```

### Ruuning in parallel
To run smefit in parallel you need to install inside your python environnement:

```bash
openmpi = 4.1.4 (4.0.2)
mpi4py = 3.1.3 (3.0.3)
```

with python 3.10 (3.9). Then you can run doing:

```bash
mpiexec -n number_of_cores python -n NS -f your_runcard_name
```

## Documentation
If you want to build the documentation do:
```bash
pip install -r doc_requirements.txt
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
## Citation policy
Please cite our DOI when using our code:
