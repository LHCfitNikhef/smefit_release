#! /bin/bash

echo
echo '                Welcome to                 '
echo '                                           '
echo '  #####  #     # ####### #######   ####### '
echo ' #     # ##   ## #       #       #    #    '
echo ' #       # # # # #       #            #    '
echo '  #####  #  #  # #####   #####   #    #    '
echo '       # #     # #       #       #    #    '
echo ' #     # #     # #       #       #    #    '
echo '  #####  #     # ####### #       #    #    '
echo '                                           '
echo '                                           '

# are you sure question
asksure() {
  echo -n "Do you want to continue (Y/n)? "
  while read -r -n 1 -s answer; do
    if [[ $answer = [YyNn] ]]; then
      [[ $answer = [Yy] ]] && retval=0
      [[ $answer = [Nn] ]] && retval=1
      break
    fi
  done
  echo
  return $retval
}


# read env name from std
while getopts :n: flag
do
    case "${flag}" in
        n) CONDA_ENV=${OPTARG};;
        *) usage ;;
    esac
done
if [ -z "${CONDA_ENV}" ];
then
  echo 'Installing to conda environenment:  smefit_installation'
  echo 'To change environenment run:'
  echo '    ./install.sh -n <env_name>'
  if ! asksure;
  then
    echo 'Quitting ...'
    exit
  else
    CONDA_ENV='smefit_installation'
  fi
else
  echo 'Installing to conda environenment: '$CONDA_ENV
fi

# select the lockfile
LOCK_FILE='conda-linux-64.lock'
if [[ $OSTYPE == 'darwin'* ]];
then
  LOCK_FILE='conda-osx-64.lock'
fi


# build the enivironment
conda create --name $CONDA_ENV --file './../conda_bld/'$LOCK_FILE
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
poetry update
poetry install

# install Multinest
MULTINEST_INSTALLATION_PATH=$PWD/'multinest_bld'
mkdir -p $MULTINEST_INSTALLATION_PATH && cd $_
git clone https://github.com/farhanferoz/MultiNest.git
cd $MULTINEST_INSTALLATION_PATH'/MultiNest/MultiNest_v3.12_CMake/multinest'
mkdir 'build' && cd $_
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
cmake ..  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
rm -rf $MULTINEST_INSTALLATION_PATH

echo
echo 'Installation was successful !!'
echo
echo 'To start type:'
echo
echo '    conda activate '$CONDA_ENV
echo '    smefit -h'
echo
