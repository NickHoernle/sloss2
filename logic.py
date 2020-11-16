from utils import *
import torch.distributions as distrib

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


alpha = 1/10
nc = 10
mu1 = np.log(alpha) - 1/nc*nc*np.log(alpha)
sigma1 = 1./alpha*(1-2./nc) + 1/(nc**2)*nc/alpha
inv_sigma1 = 1./sigma1
log_det_sigma = nc*np.log(sigma1)


class DecoderModel(nn.Module):
    def __init__(self, num_classes=10, z_dim=2, device="cpu"):
        super().__init__()

        self.nc = num_classes
        self.device = device
        self.z = z_dim

        # local parameters
        self.local_mu = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, z_dim)
        )

        self.local_lv = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, z_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, 250),
            nn.LeakyReLU(.2),
            nn.Linear(250, 100),
            nn.LeakyReLU(.2),
            nn.Linear(100, num_classes)
        )

        # shared parameters
        # self.global_mus = nn.Parameter(torch.randn(num_classes, z_dim), requires_grad=True)
        # self.global_lvs = nn.Parameter(torch.ones(1), requires_grad=True)

        self.apply(init_weights)

    def local_parameters(self):
        return [p for k, p in self.named_parameters() if ("local_mu" in k) or ("local_lv" in k) or ("net" in k)]

    # def global_parameters(self):
    #     return [p for k, p in self.named_parameters() if k == "global_mus" or k == "global_lvs"]

    def encode(self, enc):
        return self.local_mu(enc), self.local_lv(enc)

    def forward(self, x):
        n_batch = x.size(0)

        # mu2 = self.global_mus.unsqueeze(0).repeat(len(x), 1, 1)
        # lv2 = torch.ones_like(mu2) * self.global_lvs

        latent_params = self.encode(x)
        mu, lv = latent_params

        # Re-parametrize a Normal distribution
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(lv.shape[1]))
        sigma = torch.exp(0.5 * lv)

        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch,)).to(self.device)) + mu

        # z = z_0.unsqueeze(1).repeat(1, self.nc, 1)
        # probs = log_normal(z, mu2, lv2)
        # probs = -(z - mu2).pow(2).sum(dim=-1)
        probs = self.net(z_0)

        # mu1 = mu.unsqueeze(1).repeat(1, self.nc, 1)
        # lv1 = lv.unsqueeze(1).repeat(1, self.nc, 1)

        # kl_div = 0.5*((lv2-lv1) + (lv1.exp() + (mu1-mu2).pow(2))/(lv2.exp()) - 1).sum(dim=-1)
        kl_div = 0.5 * (np.log(9) - lv + (lv.exp() + mu.pow(2)) / 9 - 1)

        return probs, (kl_div, mu, lv), z_0

    def test(self, x):
        # g_mus = self.global_mus.unsqueeze(0).repeat(len(x), 1, 1)
        # g_lv = torch.ones_like(g_mus) * self.global_lvs

        latent_params = self.encode(x)
        mu, lv = latent_params

        # z_ = mu.unsqueeze(1).repeat(1, self.nc, 1)
        # probs = log_normal(z_, g_mus, g_lv)
        probs = self.net(mu)

        return probs

    # def sample(self, num_samples=1000):
    #     mus = self.global_mus.unsqueeze(0).repeat(num_samples, 1, 1)
    #     lv = torch.ones_like(mus) * self.global_lvs
    #
    #     targets = torch.arange(self.nc).repeat(num_samples // self.nc).to(self.device)
    #     z = reparameterise(mus, lv)[np.arange(num_samples), targets]
    #
    #     z_ = z.unsqueeze(1).repeat(1, self.nc, 1)
    #     probs = log_normal(z_, mus, lv)
    #     return (probs, z, targets), (self.global_mus, self.global_lvs)


class LogicNet(nn.Module):

    def __init__(self, num_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )
        self.apply(init_weights)

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


# Step 3: use logic to train encoders to map to this space. (Sampling from the posterior will ensure the
# learnt encodings are robust to the geometry that is defined on the space.)
def calc_logic_loss(probs, logic_net):
    # airplane
    pred = logic_net(probs).squeeze(1)
    true = (((probs[:, 0] + probs[:, 2] > .95)) |
            ((probs[:, 1] + probs[:, 9] > .95)) |
            ((probs[:, 3] + probs[:, 5] > .95)) |
            ((probs[:, 4] + probs[:, 7] > .95)) |
            ((probs[:, 8] + probs[:, 0] > .95)) |
            (probs[:, 6] > .95))

    return pred, true

