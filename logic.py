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


class LogicNet(nn.Module):

    def __init__(self, num_categories):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_categories, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def cifar10_logic(predictions, **kwargs):
    logic = ((predictions > 0.99) | (predictions < 0.01)).all(dim=1)
    return logic.float()

def cifar100_logic(predictions, superclass_indexes):

    superclass_predictions = torch.cat([predictions[:, superclass_indexes[c]].logsumexp(dim=1).unsqueeze(1)
                                        for c in range(len(super_class_label))], dim=1).exp()

    logic = ((superclass_predictions < 0.05) | (superclass_predictions > 0.95)).all(dim=1)

    return logic.float()