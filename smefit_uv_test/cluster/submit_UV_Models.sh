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
    WALLTIME='04:00:00'

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
    MEM=$(($NCORES*16))gb
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=$NCORES -l vmem=$MEM -l walltime=$WALLTIME $COMMAND



    # cleaning
    rm $COMMAND


}


function submit_ns () {
    NCORES='8'

    #MODELS=('2' '5' '6' '22' '23' '24' '37' '38' '39' '40' '41' '42' '43' '44' '46' '47' '48' '49' '50' '52')
    #MODELS=('Q1Q7W1_ND')

    MASS="342"
    MODELS=('Q1_Q7_W_NoDegen')
#    for ((i=2; i<=50; i++))
#    do
#      MODELS+=("$i")
#    done

    COLLECTIONS=('MultiParticleCollection')
    EFTS=('NHO' 'HO')
    PTOS=('NLO' 'LO')
    MODE='NS'
    IS_UV=true

    for MOD in ${MODELS[@]}
        do
        for PTO in ${PTOS[@]}
            do
            for EFT in ${EFTS[@]}
                do
                  for COLL in ${COLLECTIONS[@]}
                  do
                    RUNCARD_NAME=$COLL'_UV_'$MOD'_'$PTO'_'$EFT'_'$MODE
                    write_run_card $MOD $EFT $PTO $MODE $MASS $COLL

                    if [ $exit_status -eq 0 ]; then
                      echo "The Python script ran without exceptions."
                    else
                      echo "The Python script encountered an error or exception."
                      continue
                    fi
                    submit_job $MODE $RUNCARD_NAME $NCORES
                  done
                done
            done
        done
}


################################################################
############                MAIN                ################
################################################################

submit_ns
