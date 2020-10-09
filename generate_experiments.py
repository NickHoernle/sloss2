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
             f"--depth 28 --width 2 --ngpu 1 --dataroot {DATA_HOME}/data --epochs 250 "
             f"--num_labelled 4000 --cuda --ssl --lp --epoch_step [60,120,160]")

repeats = 1

dataset = ["cifar10"]
learning_rate = [.1]
unl_weight = [.1, .05, 0.01]
unl2_weight = [1., .1, .05, 0.01]
lr_decay_ratio = [.2]
num_hidden = [10]

settings = [(lr, unl_, unl2_, lr_decay_ratio_, num_hidden_, dataset_, rep)
            for lr in learning_rate
            for unl_ in unl_weight
            for unl2_ in unl2_weight
            for lr_decay_ratio_ in lr_decay_ratio
            for num_hidden_ in num_hidden
            for dataset_ in dataset
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, unl_, unl2_, lr_decay_ratio_, num_hidden_, dataset_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--unl_weight {unl_} "
        f"--unl2_weight {unl2_} "
        f"--lr_decay_ratio {lr_decay_ratio_} "
        f"--dataset {dataset_} "
        f"--num_hidden {num_hidden_} "
    )
    print(expt_call, file=output_file)

baseline = f"{base_call} --lr 0.1 --unl_weight 0 --unl2_weight .5 --lr_decay_ratio 0.7 --dataset cifar10"
print(baseline, file=output_file)

output_file.close()