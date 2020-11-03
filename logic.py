from utils import *

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
global sc_map

sc_prev = ""
count = 0

for fc, sc in sorted(superclass_mapping.items(), key=lambda x: x[1]):
    if sc == sc_prev:
        fc_mapping[fc] = count
        count += 1
    else:
        count = 0
        sc_prev = sc
        fc_mapping[fc] = count
        count += 1


def set_class_mapping(classes):
    global sc_map
    sc_map_ = np.array([super_class_label[superclass_mapping[c]] for c in classes])
    sc_map = np.argsort(sc_map_)


class DecoderModel(nn.Module):
    def __init__(self, num_classes=10, zdim=2, device="cpu"):
        super().__init__()

        self.mu = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, zdim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, zdim)
        )

        self.mu_oh = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, zdim)
        )

        self.logvar_oh = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, zdim)
        )

        self.net = nn.Sequential(
            nn.Linear(zdim, 200),
            nn.LeakyReLU(.2),
            nn.Linear(200, 500),
            nn.LeakyReLU(.2),
            nn.Linear(500, 200),
            nn.LeakyReLU(.2),
            nn.Linear(200, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, num_classes)
        )

        self.nc = num_classes
        self.nz = zdim
        self.device = device

    def get_local_params(self):
        return [p for k, p in self.named_parameters() if ("mu" in k) or ("logvar" in k)]

    def re_init_local(self):
        self.mu.apply(init_weights)
        self.logvar.apply(init_weights)

    def forward_oh(self, x):
        mu = self.mu_oh(x)
        logvar = self.logvar_oh(x)

        z = reparameterise(mu, logvar)

        return self.net(z), (z, mu, logvar)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)

        z = reparameterise(mu, logvar)

        return self.net(z), (z, mu, logvar)

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.nz).to(self.device)
        return self.net(z)


class LogicNet(nn.Module):

    def __init__(self, num_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_true_cifar100_sc(fc_labels, classes):
    return torch.tensor([super_class_label[superclass_mapping[classes[c]]] for c in fc_labels])


def get_true_cifar100_from_one_hot(one_hot):
    return torch.stack(one_hot[:, sc_map].split(5, dim=-1), dim=1).sum(dim=-1)


def get_cifar100_unnormed_pred(samples):
    return torch.stack(samples[:, sc_map].split(5, dim=-1), dim=1).logsumexp(dim=-1)


def get_cifar100_pred(samples):
    return torch.stack(samples[:, sc_map].split(5, dim=-1), dim=1).exp().sum(dim=-1)


def cifar100_logic(log_prob):

    terms = ((log_prob[:, (0, 1)].exp().sum(dim=1) > .95) |
             (log_prob[:, (1, 9)].exp().sum(dim=1) > .95) |
             (log_prob[:, (2, 0)].exp().sum(dim=1) > .95) |
             (log_prob[:, (3, 5)].exp().sum(dim=1) > .95) |
             (log_prob[:, (4, 5)].exp().sum(dim=1) > .95) |
             (log_prob[:, (5, 3)].exp().sum(dim=1) > .95) |
             (log_prob[:, 6].exp() > .95) |
             (log_prob[:, (7, 4)].exp().sum(dim=1) > .95) |
             (log_prob[:, 8].exp() > .95) |
             (log_prob[:, (9, 1)].exp().sum(dim=1) > .95))

    return terms

# def cifar100_logic_val(log_prob, labels):
#
#     mapping = {
#         0: 1,
#         1: 9,
#         2: 0,
#         3: 5,
#         4: 5,
#         5: 3,
#         6: 6,
#         7: 4,
#         8: 8,
#         9: 1,
#     }
#
#     labels_ = labels.cpu().detach().numpy()
#     labels2 = np.stack((labels_, [mapping[l] for l in labels_]), axis=1)
#     pred = torch.argmax(log_prob, dim=1).cpu().detach().numpy()
#     return (pred.reshape(-1,1) == labels2).any(axis=1)