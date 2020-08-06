import torch
from torch import nn
import torch.nn.functional as F


superclass_mapping = {
    'beaver': 'aquatic mammals',
    'dolphin': 'aquatic mammals',
    'otter': 'aquatic mammals',
    'seal': 'aquatic mammals',
    'whale': 'aquatic mammals',
    'aquarium_fish': 'fish',
    'flatfish': 'fish',
    'ray': 'fish',
    'shark': 'fish',
    'trout': 'fish',
    'orchid': 'flowers',
    'poppy': 'flowers',
    'rose': 'flowers',
    'sunflower': 'flowers',
    'tulip': 'flowers',
    'bottle': 'food containers',
    'bowl': 'food containers',
    'can': 'food containers',
    'cup': 'food containers',
    'plate': 'food containers',
    'apple': 'fruit and vegetables',
    'mushroom': 'fruit and vegetables',
    'orange': 'fruit and vegetables',
    'pear': 'fruit and vegetables',
    'sweet_pepper': 'fruit and vegetables',
    'clock': 'household electrical devices',
    'keyboard': 'household electrical devices',
    'lamp': 'household electrical devices',
    'telephone': 'household electrical devices',
    'television': 'household electrical devices',
    'bed': 'household furniture',
    'chair': 'household furniture',
    'couch': 'household furniture',
    'table': 'household furniture',
    'wardrobe': 'household furniture',
    'bee':'insects',
    'beetle':'insects',
    'butterfly':'insects',
    'caterpillar':'insects',
    'cockroach':'insects',
    'bear': 'large carnivores',
    'leopard': 'large carnivores',
    'lion': 'large carnivores',
    'tiger': 'large carnivores',
    'wolf': 'large carnivores',
    'bridge': 'large man-made outdoor things',
    'castle': 'large man-made outdoor things',
    'house': 'large man-made outdoor things',
    'road': 'large man-made outdoor things',
    'skyscraper': 'large man-made outdoor things',
    "cloud": "large natural outdoor scenes",
    "forest": "large natural outdoor scenes",
    "mountain": "large natural outdoor scenes",
    "plain": "large natural outdoor scenes",
    "sea": "large natural outdoor scenes",
    "camel": "large omnivores and herbivores",
    "cattle": "large omnivores and herbivores",
    "chimpanzee": "large omnivores and herbivores",
    "elephant": "large omnivores and herbivores",
    "kangaroo": "large omnivores and herbivores",
    "fox": "medium-sized mammals",
    "porcupine": "medium-sized mammals",
    "possum": "medium-sized mammals",
    "raccoon": "medium-sized mammals",
    "skunk": "medium-sized mammals",
    "crab": "non-insect invertebrates",
    "lobster": "non-insect invertebrates",
    "snail": "non-insect invertebrates",
    "spider": "non-insect invertebrates",
    "worm": "non-insect invertebrates",
    "baby": "people",
    "boy": "people",
    "girl": "people",
    "man": "people",
    "woman": "people",
    "crocodile" : "reptiles",
    "dinosaur" : "reptiles",
    "lizard" : "reptiles",
    "snake" : "reptiles",
    "turtle": "reptiles",
    "hamster": "small mammals",
    "mouse": "small mammals",
    "rabbit": "small mammals",
    "shrew": "small mammals",
    "squirrel": "small mammals",
    "maple_tree" :"trees",
    "oak_tree" :"trees",
    "palm_tree" :"trees",
    "pine_tree" :"trees",
    "willow_tree" :"trees",
    "bicycle": "vehicles 1",
    "bus": "vehicles 1",
    "motorcycle": "vehicles 1",
    "pickup_truck": "vehicles 1",
    "train": "vehicles 1",
    "lawn_mower": "vehicles 2",
    "rocket": "vehicles 2",
    "streetcar": "vehicles 2",
    "tank": "vehicles 2",
    "tractor": "vehicles 2"
}

super_class_label = {
    'aquatic mammals': 0,
    'fish': 1,
    'flowers': 2,
    'food containers': 3,
    'fruit and vegetables': 4,
    'household electrical devices': 5,
    'household furniture': 6,
    'insects': 7,
    'large carnivores': 8,
    'large man-made outdoor things': 9,
    'large natural outdoor scenes': 10,
    'large omnivores and herbivores': 11,
    'medium-sized mammals': 12,
    'non-insect invertebrates': 13,
    'people': 14,
    'reptiles': 15,
    'small mammals': 16,
    'trees': 17,
    'vehicles 1': 18,
    'vehicles 2': 19
}

fc_mapping = {}

sc_prev = ""
count = 0

for fc, sc in sorted(superclass_mapping.items(), key=lambda x: x[1]):

    fc_mapping[fc] = count
    count += 1

    if sc != sc_prev:
        if count > 1:
            count = 0
        sc_prev = sc


def cifar10_logic(variables, device):
    # we are dealing with one-hot assigments
    assignments = torch.eye(10).to(device)
    lower_triang = torch.tril(torch.ones_like(assignments)) - assignments

    log_probabilities = F.logsigmoid(variables)
    log_probabilities = torch.cat((log_probabilities, torch.zeros_like(log_probabilities[:, 0]).unsqueeze(1)), dim=1)

    weight = log_probabilities.unsqueeze(1) * assignments.unsqueeze(0).repeat(log_probabilities.shape[0], 1, 1)
    weight2 = torch.log(1 - torch.exp(log_probabilities)
                        .unsqueeze(1) * lower_triang.unsqueeze(0).repeat(log_probabilities.shape[0], 1, 1))
    log_WMC = (weight.sum(dim=2) + weight2.sum(dim=2))

    return log_WMC


def cifar100_logic(variables, device):
    # we are dealing with one-hot assigments
    sc_assign = torch.eye(20).to(device)
    fc_assign = torch.eye(5).to(device)

    lower_triang_sc = torch.tril(torch.ones_like(sc_assign)) - sc_assign
    lower_triang_fc = torch.tril(torch.ones_like(fc_assign)) - fc_assign

    log_probabilities = F.logsigmoid(variables)

    sc_predictions = log_probabilities[:, :19]
    fc_predictions = log_probabilities[:, 19:]

    sc_log_prob = torch.cat((sc_predictions, torch.zeros_like(sc_predictions[:, 0]).unsqueeze(1)), dim=1)
    fc_log_prob = torch.cat((fc_predictions, torch.zeros_like(fc_predictions[:, 0]).unsqueeze(1)), dim=1)

    weight_sc = sc_log_prob.unsqueeze(1) * sc_assign.unsqueeze(0).repeat(sc_log_prob.shape[0], 1, 1)
    weight2_sc = torch.log(1 - torch.exp(sc_log_prob).unsqueeze(1) *
                           lower_triang_sc.unsqueeze(0).repeat(sc_log_prob.shape[0], 1, 1))

    weight_fc = fc_log_prob.unsqueeze(1) * fc_assign.unsqueeze(0).repeat(fc_log_prob.shape[0], 1, 1)
    weight2_fc = torch.log(1 - torch.exp(fc_log_prob).unsqueeze(1) *
                           lower_triang_fc.unsqueeze(0).repeat(fc_log_prob.shape[0], 1, 1))

    log_WMC_sc = (weight_sc.sum(dim=2) + weight2_sc.sum(dim=2)).repeat(1, 5)
    log_WMC_fc = (weight_fc.sum(dim=2) + weight2_fc.sum(dim=2)).view(-1, 1).repeat(1, 20).view(-1, 100)
    log_WMC = log_WMC_sc + log_WMC_fc

    return log_WMC