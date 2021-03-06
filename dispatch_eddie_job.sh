#!/usr/bin/env bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=01:00:00
#$ -N zsemloss
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Minimum 16 GB RAM for K80 GPUs
#$ -pe gpu 1
#$ -l h_vmem=32G



# Initialise the environment modules and load CUDA version 8.0.61
#. /etc/profile.d/modules.sh
source /exports/applications/support/set_cuda_visible_devices.sh
module load anaconda

# Run the executable

git_commit="`git rev-parse HEAD`"
echo "Last git commit: ${git_commit}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/exports/eddie/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=sloss
echo "Activating conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME}

# ensure you are running the latest code
#pip uninstall -y semantic-loss
#pip install .

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/exports/csce/eddie/inf/groups/cisa_gal/nick/git/semantic_loss
src_path=${repo_home}/data/vaelib
dest_path=${SCRATCH_DISK}/${USER}/vaelib
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}/data

num_lines=$(ls -l ${dest_path}/* | wc -l)
echo "Number of files at the destination: ${num_lines}"

input_dir=${dest_path}/data
output_dir=${dest_path}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/models
mkdir -p ${output_dir}/logs

cd /exports/csce/eddie/inf/groups/cisa_gal/nick/git/sloss2

#experiment_text_file=$1
#COMMAND="`sed \"${SGE_TASK_ID}q;d\" ${experiment_text_file}`"
COMMAND="python main.py --save ${SCRATCH_HOME}/vaelib/logs/resnet_$RANDOM$RANDOM --depth 28 --width 2 --ngpu 1 --dataroot ${SCRATCH_HOME}/vaelib/data --num_labelled 4000 --cuda --ssl --lp --lr 0.1 --unl_weight 0.1 --num_hidden 10 --lr_decay_ratio 0.7 --dataset cifar10 --epoch_step [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"