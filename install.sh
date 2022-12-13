#! /bin/bash
CONDA_ENV='smefit_installation'


while getopts :p: flag
do
    case "${flag}" in
        p) MULTINEST_INSTALLATION_PATH=${OPTARG};;
        *) usage ;;
    esac
done

if [ -z "${MULTINEST_INSTALLATION_PATH}" ];
then
    echo "Usage: $0 -p MULTINEST_INSTALLATION_PATH"
    exit 1
fi


# select the lockfile
LOCK_FILE='conda-linux-64.lock'
if [[ $OSTYPE == 'darwin'* ]];
then
  LOCK_FILE='conda-osx-64.lock'
fi


# build the enivironment
conda create --name $CONDA_ENV --file 'conda_bld/'$LOCK_FILE
conda activate $CONDA_ENV
poetry install

# install Multinest
cd $MULTINEST_INSTALLATION_PATH
mkdir build/
cd build/
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
cmake ..  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
