#! /bin/bash

usage() { echo "Usage: $0 -p MULTINEST_INSTALLATION_PATH"; exit 1; }

while getopts :p: flag
do
    case "${flag}" in
        p) MULTINEST_INSTALLATION_PATH=${OPTARG};;
    esac
done

if [ -z "${path}" ]; then
    usage
fi


# TODO: try to detect if multinest is already installed and skip


# TODO: check if on MacOS this works
# get openmipi lib dev
sudo apt install libopenmpi-dev

# install Multinest
git clone https://github.com/farhanferoz/MultiNest.git
cd MultiNest/MultiNest_v3.12_CMake/multinest
mkdir build && cd $_
cmake .. -DCMAKE_INSTALL_PREFIX=$MULTINEST_INSTALLATION_PATH
make
sudo make install
cd ../../../../

# upgrade pip
python -m pip install --upgrade pip

# install smefit
pip install packutil
pip install .
