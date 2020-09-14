#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/vaelib'
base_call = (f"python main.py --save {DATA_HOME}/logs/resnet_$RANDOM$RANDOM "
             f"--depth 28 --width 2 --ngpu 1 --dataroot {DATA_HOME}/data "
             f"--num_labelled 4000 --cuda --ssl --lp")

repeats = 1

dataset = ["cifar10"]
learning_rate = [5e-1, 1e-1, 5e-2]
unl_weight = [1, .5, 0.1]
lr_decay_ratio = [0.2]

settings = [(lr, unl_, lr_decay_ratio_, dataset_, rep)
            for lr in learning_rate
            for unl_ in unl_weight
            for lr_decay_ratio_ in lr_decay_ratio
            for dataset_ in dataset
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, unl_, lr_decay_ratio_, dataset_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--unl_weight {unl_} "
        f"--lr_decay_ratio {lr_decay_ratio_} "
        f"--dataset {dataset_} "
    )
    print(expt_call, file=output_file)

output_file.close()