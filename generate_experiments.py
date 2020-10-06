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
             f"--depth 28 --width 2 --ngpu 1 --dataroot {DATA_HOME}/data --epochs 300 "
             f"--num_labelled 4000 --cuda --ssl --lp --epoch_step [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]")

repeats = 1

dataset = ["cifar10"]
learning_rate = [.1]
unl_weight = [1., .1, .01]
unl2_weight = [.1, .05, .01]
lr_decay_ratio = [.7]

settings = [(lr, unl_, unl2_, lr_decay_ratio_, dataset_, rep)
            for lr in learning_rate
            for unl_ in unl_weight
            for unl2_ in unl2_weight
            for lr_decay_ratio_ in lr_decay_ratio
            for dataset_ in dataset
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, unl_, unl2_, lr_decay_ratio_, dataset_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--unl_weight {unl_} "
        f"--unl2_weight {unl2_} "
        f"--lr_decay_ratio {lr_decay_ratio_} "
        f"--dataset {dataset_} "
    )
    print(expt_call, file=output_file)

baseline = f"{base_call} --lr 0.1 --unl_weight 0 --unl2_weight 1. --lr_decay_ratio 0.7 --dataset cifar10"
print(baseline, file=output_file)

output_file.close()