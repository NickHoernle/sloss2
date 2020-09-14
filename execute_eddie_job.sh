#!/usr/bin/env bash

EXPT_FILE=experiment.txt
NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
MAX_PARALLEL_JOBS=20
qsub -t 1-${NR_EXPTS} dispatch_eddie_job.sh $EXPT_FILE