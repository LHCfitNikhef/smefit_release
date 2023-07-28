# bin/bash

ENV='smefit_installation'
source ./utils.sh

function write_run_card () {
    cd ..

    if [ "$IS_UV" == true  ]
        then
        RUNCARD_NAME=$6'_UV_'$1'_'$3'_'$2'_'$4
        RUNCARD_FOLDER='uv_models'
        else
        RUNCARD_NAME=$6'_'$1'_'$3'_'$2'_'$4
        RUNCARD_FOLDER='WC_models'
    fi
    $PY $PWD'/runcards/'$RUNCARD_FOLDER'/write_runcards.py' '-i'$1 '-e'$2 '-o'$3 '-f'$4 '-m'$5 '-c'$6 '-u'
    exit_status=$?
    cd 'cluster'
}


function submit_job () {

    # FIT SETUP
    MODE=$1
    FIT_ID=$2
    NCORES=$3
    MASS=$4
    WALLTIME='12:00:00'

    # create the bash file to submit
    COMMAND=$PWD'/launch_'$FIT_ID'.sh'
    cd ..
    ROOT_PATH=$PWD
    cd $PWD'/cluster'
    OUT_PATH=$PWD'/logs'

    # this is the script to launch
    RUNCARD_PATH=$ROOT_PATH'/runcards'
    RUNCARD=$RUNCARD_PATH'/'$FIT_ID'.yaml'

    EXPORT='export LD_LIBRARY_PATH='$MULTINEST';'

    if [ $MODE == 'MC' ]
    then
        NREP=$4
        OUT_PATH=$OUT_PATH'/mc_logs'
        SMEFIT_ARGS='MC '$RUNCARD' -n '$NREP' -l '$OUT_PATH'/output_'$FIT_ID'.log;'
        EXE=$SMEFIT' '$SMEFIT_ARGS
    fi
    if [ $MODE == 'NS' ]
    then
        SMEFIT_ARGS='NS '$RUNCARD' -l '$OUT_PATH'/output_'$FIT_ID'.log;'
        EXE=$MPI' -n '$NCORES' '$SMEFIT' '$SMEFIT_ARGS
    fi

    LAUNCH=$EXPORT$EXE

    [ -e $COMMAND ] && rm $COMMAND
    mkdir -p $OUT_PATH
    echo $LAUNCH >> $COMMAND
    chmod +x $COMMAND

    # submission
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=$NCORES -l vmem=128gb -l walltime=$WALLTIME $COMMAND



    # cleaning
    rm $COMMAND


}

function submit_job_sequential () {
    # FIT SETUP
    FIT_ID=$1
    NCORES=$2
    REP_IN=$3
    REP_MAX=$4

    # create the bash file to submit
    COMMAND=$PWD'/launch_'$FIT_ID'.sh'
    cd ..
    ROOT_PATH=$PWD
    cd $PWD'/cluster'
    OUT_PATH=$PWD'/logs/mc_logs'

    # this is the script to launch
    RUNCARD_PATH=$ROOT_PATH'/runcards'
    RUNCARD=$RUNCARD_PATH'/'$FIT_ID'.yaml'

    EXPORT='export LD_LIBRARY_PATH='$MULTINEST';'
    SMEFIT_ARGS='MC '$RUNCARD' -n $NREP -l '$OUT_PATH'/output_'$FIT_ID'.log;'
    EXE=$SMEFIT' '$SMEFIT_ARGS
    LAUNCH=$EXPORT'REPLIST=($(seq '$REP_IN' 1 '$REP_MAX')); for NREP in ${REPLIST[@]};do '$EXE'done;'

    [ -e $COMMAND ] && rm $COMMAND
    mkdir -p $OUT_PATH
    echo $LAUNCH >> $COMMAND
    chmod +x $COMMAND

    echo $COMMAND

    # submission
    qsub -q short7 -W group_list=theorie -l nodes=1:ppn=$NCORES -l pvmem=8000mb -l walltime=12:00:00 $COMMAND

    # cleaning
    rm $COMMAND
}

function submit_ns () {
    NCORES='8'

    #MODELS=('2' '5' '6' '22' '23' '24' '37' '38' '39' '40' '41' '42' '43' '44' '46' '47' '48' '49' '50' '52')
    #MODELS=('Q1Q7W1_ND')

    MASS="free"
    MODELS=('5')
#    for ((i=2; i<=50; i++))
#    do
#      MODELS+=("$i")
#    done

    EFTS=('NHO' 'HO')
    PTOS=('LO' 'NLO')
    MODE='NS'
    IS_UV=true

    for MOD in ${MODELS[@]}
        do
        for PTO in ${PTOS[@]}
            do
            for EFT in ${EFTS[@]}
                do
                RUNCARD_NAME='1L_UV_'$MOD'_'$PTO'_'$EFT'_'$MODE'_'$MASS
                write_run_card $MOD $EFT $PTO $MODE $MASS "1L"

                if [ $exit_status -eq 0 ]; then
                  echo "The Python script ran without exceptions."
                else
                  echo "The Python script encountered an error or exception."
                  continue
                fi
                submit_job $MODE $RUNCARD_NAME $NCORES $MASS
                done
            done
        done
}


################################################################
############                MAIN                ################
################################################################

submit_ns
