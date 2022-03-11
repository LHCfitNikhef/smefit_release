# /bin/bash

# get openmipi lib dev
sudo apt install libopenmpi-dev

# install Multinest
git clone https://github.com/farhanferoz/MultiNest.git
cd MultiNest/MultiNest_v3.12_CMake/multinest
mkdir build && cd $_
cmake ..
make

# upgrade pip
python -m pip install --upgrade pip

# install smefit
python -m pip install packutil
python setup.py install
