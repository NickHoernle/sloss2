import torch
import random
import os
from torch import nn
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
import numpy as np
from datasets import get_CIFAR10, get_CIFAR100


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j * len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True


def x_u_split(labels, num_labelled, num_classes):
    label_per_class = num_labelled // num_classes
    labels = np.array(labels)
    labelled_idx = []
    unlabelled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labelled_idx.extend(idx[:label_per_class])
        unlabelled_idx.extend(idx[label_per_class:])

    return labelled_idx, unlabelled_idx


def calculate_accuracy(preds, labels):
    no_examples = labels.shape[0]
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    preds = (preds > 0.5).astype(int)
    acc = np.sum(np.all(preds == labels, axis=1)) / float(labels.shape[0])
    return acc * 100


def one_hot_embedding(labels, num_classes, device="cuda:0"):
    y = torch.eye(num_classes).to(device)
    return y[labels]


def log_normal(x, m, log_v):
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    log_det = -0.5 * torch.sum(log_v, dim=-1)
    log_exp = -0.5 * torch.sum((x - m) ** 2 / (log_v.exp()), dim=-1)

    log_prob = const + log_det + log_exp

    return log_prob


def check_dataset(dataset, dataroot, download):
    if dataset == "cifar10":
        return get_CIFAR10(dataroot, download)
    if dataset == "cifar100":
        return get_CIFAR10(dataroot, download)

    raise NotImplementedError(f"No dataset for {dataset}")


def check_manual_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reparameterise(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)