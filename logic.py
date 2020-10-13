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
    def __init__(self, num_classes, z_dim=2):
        super().__init__()

        # local params
        self.mu = nn.Sequential(nn.Linear(num_classes, 50), nn.LeakyReLU(.2), nn.Linear(50, z_dim))
        self.logvar = nn.Sequential(nn.Linear(num_classes, 50), nn.LeakyReLU(.2), nn.Linear(50, z_dim))
        self.log_pi = nn.Sequential(nn.Linear(num_classes, 50), nn.LeakyReLU(.2), nn.Linear(50, num_classes))

        # global params
        self.cluster_means = nn.Parameter(torch.randn(num_classes, z_dim), requires_grad=True)
        self.cluster_lvariances = nn.Parameter(torch.zeros(num_classes, z_dim), requires_grad=True)

        self.net = nn.Sequential(nn.Linear(z_dim, 50), nn.LeakyReLU(.2), nn.Linear(50, num_classes))

        self.nc = num_classes
        self.zdim = z_dim
        self.apply(init_weights)

    def get_global_params(self):
        return [v for k, v in self.named_parameters() if ("cluster_means" in k) or ("cluster_lvariances" in k)]

    def get_local_params(self):
        return [v for k, v in self.named_parameters() if ("mu" in k) or ("logvar" in k)]

    def forward(self, x):
        # encode
        mu = self.mu(x)
        logvar = self.logvar(x)

        # resample
        z = reparameterise(mu, logvar)

        # evaluate cluster params
        cluster_mus = self.cluster_means.unsqueeze(0).repeat(len(x), 1, 1)
        cluster_logvars = torch.zeros_like(cluster_mus)

        # calculate log-prob
        prediction = self.net(z)

        return prediction, (z, mu, logvar, cluster_mus, cluster_logvars)

    def train_generative_only(self, num_samples):
        cluster_mus = self.cluster_means.unsqueeze(0).repeat(num_samples, 1, 1)
        cluster_logvars = torch.zeros_like(cluster_mus)

        z2 = reparameterise(cluster_mus, cluster_logvars)

        log_probs = torch.stack([self.net(z2[:, i, :]) for i in range(self.nc)], dim=1)
        return log_probs, (cluster_mus, cluster_logvars)


class LogicNet(nn.Module):

    def __init__(self, num_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 25),
            nn.Sigmoid(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def get_true_cifar100_sc(fc_labels):
    return torch.tensor([sc_map[c] for c in fc_labels])


def get_cifar100_pred(samples):
    return torch.stack(samples.split(5, dim=-1), dim=1).exp().sum(dim=-1)


def cifar100_logic(samples):
    super_class_preds = get_cifar100_pred(samples)
    return ((super_class_preds > 0.95) | (super_class_preds < 0.05)).all(dim=1)