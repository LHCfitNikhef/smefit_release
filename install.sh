#! /bin/bash
CONDA_ENV='smefit_installation'
MULTINEST_INSTALLATION_PATH=$PWD/'multinest_bld'

# select the lockfile
LOCK_FILE='conda-linux-64.lock'
if [[ $OSTYPE == 'darwin'* ]];
then
  LOCK_FILE='conda-osx-64.lock'
fi


# build the enivironment
conda create --name $CONDA_ENV --file 'conda_bld/'$LOCK_FILE
conda init bash
conda activate $CONDA_ENV
poetry install

# install Multinest
cd $MULTINEST_INSTALLATION_PATH
git clone https://github.com/farhanferoz/MultiNest.git
cd MultiNest/MultiNest_v3.12_CMake/multinest
mkdir build && cd $_
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
cmake ..  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
rm -rf $MULTINEST_INSTALLATION_PATH
