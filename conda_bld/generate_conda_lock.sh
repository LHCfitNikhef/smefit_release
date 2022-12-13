#! /bin/bash
MODE=1


TMP_PATH=$PWD/'test_env'
CONDA_LOCK=$TMP_PATH'/bin/conda-lock'
MAMBA=$TMP_PATH'/bin/mamba'
POETRY=$TMP_PATH'/bin/poetry'


# conda lock generation
if [ "$MODE" -eq "1" ];
then
    conda create -p $TMP_PATH -c conda-forge mamba conda-lock poetry
    $CONDA_LOCK -f $PWD/'environment_linux.yml' -p linux-64 -k explicit --conda mamba
    $CONDA_LOCK -f $PWD/'environment_osx.yml' -p osx-64 -k explicit --conda mamba
    rm -rf $TMP_PATH
fi;

# Re-generate Conda lock file(s) based on environment.yml
if [ "$MODE" -eq "2" ];
then
    conda create -p $TMP_PATH -c conda-forge mamba conda-lock poetry
    $CONDA_LOCK -k explicit --conda mamba
    $MAMBA update --file 'conda-linux-64.lock'
    $MAMBA update --file 'conda-osx-64.lock'
    $POETRY update
    rm -rf $TMP_PATH
fi;
