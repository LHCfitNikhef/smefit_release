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


```bash
git clone https://github.com/LHCfitNikhef/smefit_release.git
cd smefit_release
./install.sh -p MULTINEST_INSTALLATION_PATH
```
The script will download and compile the MultiNest library, together with the necessary python packages
to run the code.

## Conda installation

For the conda users, the following installation steps will create a conda environment with a working version
of SmeFit installed

```bash
conda create -n <ENV_NAME> python=3.10
conda install -c conda-forge openmpi=4.1.4=ha1ae619_100
conda install mpi4py=3.1.3
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

## Running
To run the code you can do:
```bash
python MODE -f your_runcard_name
```
where MODE can be NS, MC, SCAN or PF. We refer to the documenttion for more details and tutorials

### Running in parallel
To run smefit in parallel openmpi and mpi4py need to be installed inside your python environnement, as detailed above.
Then you can run:
```bash
mpiexec -n number_of_cores python NS -f your_runcard_name
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
