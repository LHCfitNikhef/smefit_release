# /bin/bash

# get openmipi lib dev
sudo apt install libopenmpi-dev

# install Multinest
git clone https://github.com/farhanferoz/MultiNest.git
cd MultiNest/MultiNest_v3.12_CMake/multinest
mkdir build && cd $_
cmake ..
make
cd ../../../../

# upgrade pip
python -m pip install --upgrade pip

# install smefit
pip install packutil
pip install .
