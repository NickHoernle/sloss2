#!/usr/bin/env bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=01:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU:
#$ -pe gpu 1
#
# Request 4 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=4G

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
module load cuda/8.0.61

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

experiment_text_file=$1
COMMAND="`sed \"${SGE_TASK_ID}q;d\" ${experiment_text_file}`"

echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"