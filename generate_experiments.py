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
             f"--num_labelled 4000 --cuda --ssl --generative_loss --epoch_step [40,80,120,160]")

repeats = 1

experiment = "cifar10"
dataset = [experiment]
learning_rate = [.1, .05]
unl_weight = [.1, .01]
unl2_weight = [1e-2, 5e-3, 1e-3]
sloss_weight = [1.]
lr_decay_ratio = [.2]
num_hidden = [10, 2]

settings = [(lr, unl_, sloss_, unl2_, lr_decay_ratio_, num_hidden_, dataset_, rep)
            for lr in learning_rate
            for unl_ in unl_weight
            for sloss_ in sloss_weight
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

for (lr, unl_, sloss_, unl2_, lr_decay_ratio_, num_hidden_, dataset_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script

    if lr == .1 and num_hidden_ == 2:
        continue
    if lr == .01 and num_hidden_ == 10:
        continue

    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--unl_weight {unl_} "
        f"--unl2_weight {unl2_} "
        f"--sloss_weight {sloss_} "
        f"--lr_decay_ratio {lr_decay_ratio_} "
        f"--dataset {dataset_} "
        f"--num_hidden {num_hidden_} "
    )
    print(expt_call, file=output_file)

baseline = f"{base_call} --lr 0.1 --sloss_weight 0 --unl_weight .1 --lr_decay_ratio 0.7 --dataset {experiment}"
print(baseline, file=output_file)

output_file.close()