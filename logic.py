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


class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, num_dim)
        )

    def forward(self, x):
        return self.net(x)


class LogicNet(nn.Module):

    def __init__(self, num_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 25),
            nn.ReLU(True),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

for fc, sc in sorted(superclass_mapping.items(), key=lambda x: x[1]):
    if sc == sc_prev:
        fc_mapping[fc] = count
        count += 1
    else:
        count = 0
        sc_prev = sc
        fc_mapping[fc] = count
        count += 1


def log_sigmoid(x):
    return x - torch.log1p(x)


def log_sigmoid(x):
    return x - torch.log1p(x)


def cifar10_logic(variables, device):
    probs = torch.softmax(variables, dim=1)
    return ((probs > 0.95) | (probs < 0.05)).all(dim=1).float()
    # we are dealing with one-hot assigments
    # assignments = torch.eye(10).to(device)
    # lower_triang = torch.tril(torch.ones_like(assignments)) - assignments
    #
    # log_probabilities = F.logsigmoid(variables)
    # log_1_min_prob = F.logsigmoid(-variables)
    #
    # log_prob = torch.cat((log_probabilities, torch.zeros_like(log_probabilities[:, 0]).unsqueeze(1)), dim=1)
    # log_1_min_prob = torch.cat((log_1_min_prob, torch.zeros_like(log_1_min_prob[:, 0]).unsqueeze(1)), dim=1)
    #
    # weight = log_prob.unsqueeze(1) * assignments.unsqueeze(0).repeat(log_prob.shape[0], 1, 1)
    # weight2 = log_1_min_prob.unsqueeze(1) * lower_triang.unsqueeze(0).repeat(log_1_min_prob.shape[0], 1, 1)
    #
    # log_WMC = (weight.sum(dim=2) + weight2.sum(dim=2))
    #
    # return log_WMC


def cifar100_logic(variables, device):
    # we are dealing with one-hot assigments
    sc_assign = torch.eye(20).to(device)
    fc_assign = torch.eye(5).to(device)

    lower_triang_sc = torch.tril(torch.ones_like(sc_assign)) - sc_assign
    lower_triang_fc = torch.tril(torch.ones_like(fc_assign)) - fc_assign

    log_probabilities = F.logsigmoid(variables)
    log_1_min_prob = F.logsigmoid(-variables)

    sc_pred = log_probabilities[:, :19]
    fc_pred = log_probabilities[:, 19:]
    # fc_pred = log_probabilities[:, 19:].view(log_probabilities.shape[0], -1, 4)

    sc_1min_pred = log_1_min_prob[:, :19]
    fc_1min_pred = log_1_min_prob[:, 19:]
    # fc_1min_pred = log_1_min_prob[:, 19:].view(log_1_min_prob.shape[0], -1, 4)

    sc_log_prob = torch.cat((sc_pred, torch.zeros_like(sc_pred[:, 0]).unsqueeze(1)), dim=1)
    fc_log_prob = torch.cat((fc_pred, torch.zeros_like(fc_pred[:, 0]).unsqueeze(1)), dim=1)
    # fc_log_prob = torch.cat((fc_pred, torch.zeros_like(fc_pred[:, :, 0]).unsqueeze(2)), dim=2)

    sc_log_1min_prob = torch.cat((sc_1min_pred, torch.zeros_like(sc_1min_pred[:, 0]).unsqueeze(1)), dim=1)
    fc_log_1min_prob = torch.cat((fc_1min_pred, torch.zeros_like(fc_1min_pred[:, 0]).unsqueeze(1)), dim=1)
    # fc_log_1min_prob = torch.cat((fc_1min_pred, torch.zeros_like(fc_1min_pred[:, :, 0]).unsqueeze(2)), dim=2)

    weight_sc = sc_log_prob.unsqueeze(1) * sc_assign.unsqueeze(0).repeat(sc_log_prob.shape[0], 1, 1)
    weight2_sc = sc_log_1min_prob.unsqueeze(1) * lower_triang_sc.unsqueeze(0).repeat(sc_log_1min_prob.shape[0], 1, 1)

    weight_fc = fc_log_prob.unsqueeze(1) * fc_assign.unsqueeze(0).repeat(fc_log_prob.shape[0], 1, 1)
    weight2_fc = fc_log_1min_prob.unsqueeze(1) * lower_triang_fc.unsqueeze(0).repeat(fc_log_1min_prob.shape[0], 1, 1)

    # weight_fc = fc_log_prob.unsqueeze(2) * fc_assign.view(1,1,5,5).repeat(fc_log_prob.shape[0], 1, 1, 1)
    # weight2_fc = fc_log_1min_prob.unsqueeze(2) * lower_triang_fc.unsqueeze(0).repeat(fc_log_1min_prob.shape[0], 1, 1, 1)

    log_WMC_sc = (weight_sc.sum(dim=2) + weight2_sc.sum(dim=2))
    log_WMC_fc = (weight_fc.sum(dim=2) + weight2_fc.sum(dim=2))
    # log_WMC_fc = (weight_fc.sum(dim=3) + weight2_fc.sum(dim=3))

    log_WMC_sc_p = log_WMC_sc.view(-1, 1).repeat(1, 5).view(-1, 100)
    log_WMC_fc_p = log_WMC_fc.repeat(1, 20)
    # log_WMC_fc_p = log_WMC_fc.view(-1, 100)

    return log_WMC_sc, log_WMC_fc, (log_WMC_sc_p+log_WMC_fc_p)