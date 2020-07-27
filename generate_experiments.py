#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/vaelib'
base_call = (f"python main.py --dataset CIFAR10 --save {DATA_HOME}/logs/resnet_$RANDOM$RANDOM "
             f"--depth 28 --width 2 --ngpu 1 --dataroot {DATA_HOME}/data --starter_counter 10 --cuda")

repeats = 1
sloss = [True]
learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
sloss_weights = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
sloss2_weights = [1e-1]

settings = [(lr, sloss_, sloss_weight, sloss2_weight, rep)
            for lr in learning_rate
            for sloss_ in sloss
            for sloss_weight in sloss_weights
            for sloss2_weight in sloss2_weights
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, sloss_, sloss_weight, sloss2_weight, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--sloss {sloss_} "
        f"--sloss_weight {sloss_weight} "
        f"--unl_weight {sloss2_weight} "
    )
    print(expt_call, file=output_file)

output_file.close()