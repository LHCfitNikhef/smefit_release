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
## Installation from source
A the moment the code is not deployed yet, you can install it only from souce.
The easiest way to install it is by running the script `install.sh`
afret having activated your enevironment:

```bash
./install.sh -p MULTINEST_INSTALLATION_PATH
```

This will download and install the [MULtiNest](https://github.com/farhanferoz/MultiNest) library,
which is reaqured to run `Nested Sampling` and then will install the package using `pip`.

## Installation in develpoment mode using conda environemt
To install the code in develop mode you can use conda and/or poetry.
What you have to run is something similar to this code.

```bash
conda create -n <ENV_NAME> python=3.10
conda install -c conda-forge openmpi=4.1.4=ha1ae619_100
conda install compilers
conda install liblapack libblas

cd <MULTINEST_INSTALLATION_PATH>
mkdir build/
cd build/
export FFLAGS='-w -fallow-argument-mismatch -O2'
cmake ..  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install

cd <SMEFIT_INSTALLATION_PATH>

conda install poetry
poetry install

```
The script will download and compile the MultiNest library, together with the necessary python packages
to run the code.

## Running
The fitting code provide two equivalent fiining stategies.
To run the code with `Nested Sampling` you can do:

```bash
smefit NS -f <your_runcard_name>
```

To run the code suing the Monte Carlo replica method you can do:

```bash
smefit MC -f <your_runcard_name> -n <replica_number>
```

An runcard example is provided in `runcards/test_runcard.yaml`.
You can also do `smefit -h` for more help.

### Ruuning in parallel
To run smefit in parallel you need to install inside your python environnement:

```bash
openmpi = 4.1.4 (4.0.2)
mpi4py = 3.1.3 (3.0.3)
```

with python 3.10 (3.9). Then you can run doing:

### Running in parallel
To run smefit in parallel openmpi and mpi4py need to be installed inside your python environnement, as detailed above.
Then you can run:
```bash
mpiexec -n number_of_cores smefit NS -f <your_runcard_name>
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
## Citation policy
Please cite our DOI when using our code:
