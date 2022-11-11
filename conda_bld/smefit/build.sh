conda install -c conda-forge openmpi=4.1.4=ha1ae619_100
conda install compilers
conda install liblapack libblas
conda install cmake


# build multinest
mkdir multinest_build/
cd multinest_build/
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
cmake ..  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install


# build pdflatex
mkdir pdflatx_build
cd pdflatx_build
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
zcat install-tl-unx.tar.gz | tar xf -
cd install-tl-*
perl ./install-tl PREFIX=$CONDA_PREFIX --no-interaction

cd ..
conda install poetry
poetry install
