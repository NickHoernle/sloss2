import argparse
import os
import json
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch import nn

import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel, print_tensor_dict, x_u_split, calculate_accuracy
from torch.backends import cudnn
from resnet import resnet
from datasets import get_CIFAR10, get_SVHN, Joint, get_AwA2
from flows import Invertible1x1Conv, NormalizingFlowModel
from spline_flows import NSF_CL
from torch.distributions import MultivariateNormal
import itertools
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

cudnn.benchmark = True

parser = argparse.ArgumentParser()
# Model options
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--width', default=2, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[25, 50, 100]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)
parser.add_argument("--no_augment", action="store_false",
                    dest="augment", help="Augment training data")

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='.', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--download", action="store_true",
                    help="downloads dataset")

# SSL options
parser.add_argument("--ssl", action="store_true",
                    help="Do semi-supervised learning")
parser.add_argument("--num_labelled", type=int, default=4000,
                    help="Number of labelled data points")
parser.add_argument("--min_entropy", action="store_true",
                    help="Add the minimum entropy loss")
parser.add_argument("--lp", action="store_true",
                    help="Add the learned prior (LP) loss")
parser.add_argument("--semantic_loss", action="store_true",
                    help="Add the semantic loss")
parser.add_argument("--unl_weight", type=float, default=0.1,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--unl2_weight", type=float, default=0.1,
                    help="Weight for unlabelled regularizer loss")


def one_hot_embedding(labels, num_classes, device="cuda:0"):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).to(device)
    return y[labels]


def log_normal(x, m, log_v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance
    Return:
        kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
            that the summation dimension (dim=-1) is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    #print("q_m", m.size())
    #print("q_v", v.size())
    const = -0.5*x.size(-1)*torch.log(2*torch.tensor(np.pi))
    #print(const.size())
    log_det = -0.5*torch.sum(log_v, dim=-1)
    #print("log_det", log_det.size())
    log_exp = -0.5*torch.sum((x - m)**2/(log_v.exp()), dim = -1)

    log_prob = const + log_det + log_exp

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Thanks: https://github.com/divymurli/VAEs/blob/master/codebase/utils.py
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution
    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance
    Returns:z
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        dataset = get_CIFAR10(augment, dataroot, download)
    if dataset == "svhn":
        dataset = get_SVHN(augment, dataroot, download)
    if dataset == "awa2":
        dataset = get_AwA2(augment, dataroot)
    return dataset


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
    return mu + eps * std


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class DecoderModel(nn.Module):
    def __init__(self, num_classes, z_dim=2):
        super().__init__()
        self.mu_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_classes, z_dim)
        )

        self.logvar_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_classes, z_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

        self.scale_params = nn.Parameter(torch.ones(num_classes), requires_grad=True)

    def forward(self, x):
        # Compute the mixture of Gaussian prior
        # prior = gaussian_parameters(self.z_pre, dim=1)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        z = reparameterise(mu, logvar)
        # z = torch.log_softmax(z_, dim=1)

        identity = z
        output = self.net(z) + identity

        return output, mu, logvar


def main():
    device = "cuda:0"
    # device = "cpu"

    args = parser.parse_args()
    print('parsed options:', vars(args))
    epoch_step = json.loads(args.epoch_step)

    check_manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ds = check_dataset(args.dataset, args.dataroot, args.augment, args.download)

    if args.dataset == "awa2":
        image_shape, num_classes, train_dataset, test_dataset, all_labels = ds
        all_labels = all_labels.to(device)
    else:
        image_shape, num_classes, train_dataset, test_dataset = ds
        all_labels = torch.eye(num_classes).to(device)

    if args.ssl:
        num_labelled = args.num_labelled
        num_unlabelled = len(train_dataset) - num_labelled
        if args.dataset == "awa2":
            labelled_set, unlabelled_set = data.random_split(train_dataset, [num_labelled, num_unlabelled])
        else:
            td_targets = train_dataset.targets if args.dataset == "cifar10" else train_dataset.labels
            labelled_idxs, unlabelled_idxs = x_u_split(td_targets, num_labelled, num_classes)
            labelled_set, unlabelled_set = [Subset(train_dataset, labelled_idxs),
                                            Subset(train_dataset, unlabelled_idxs)]
        labelled_set = data.ConcatDataset([labelled_set for i in range(num_unlabelled // num_labelled + 1)])
        labelled_set, _ = data.random_split(labelled_set, [num_unlabelled, len(labelled_set) - num_unlabelled])

        train_dataset = Joint(labelled_set, unlabelled_set)

    def _init_fn(worker_id):
        np.random.seed(args.seed)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        worker_init_fn=_init_fn
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        worker_init_fn=_init_fn
    )
    z_dim = 10
    model, params = resnet(args.depth, args.width, num_classes, image_shape[0])

    if args.lp:
        model_y = DecoderModel(num_classes, z_dim)
        model_y.to(device)
        model_y.apply(init_weights)

    def create_optimizer(args, lr):
        print('creating optimizer with lr = ', lr)
        params_ = [v for v in params.values() if v.requires_grad]
        if args.lp:
            params_ += list(model_y.parameters())
        # return Adam(params_, lr)
        return SGD(params_, lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer(args, args.lr)

    epoch = 0

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    if args.dataset == "awa2":
        classacc = tnt.meter.AverageValueMeter()
    else:
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    global counter, aggressive
    counter = 0
    aggressive = False

    # device = torch.cuda.current_device()
    # print(f"On GPU: {device}")
    #
    # print(f"{torch.cuda.device(device)}")
    #
    # print(f"# devices: {torch.cuda.device_count()}")
    #
    # print(f"Device name: {torch.cuda.get_device_name(device)}")
    #
    # print(f"{torch.cuda.is_available()}")

    alpha = 1. / num_classes
    mu_prior = np.log(alpha) - 1 / num_classes * num_classes * np.log(alpha)
    sigma_prior = 1. / alpha * (1 - 2. / num_classes) + 1 / (num_classes ** 2) * num_classes / alpha
    prior_log_var = np.log(sigma_prior)
    inv_sigma1 = 1. / sigma_prior
    log_det_sigma = num_classes * np.log(sigma_prior)

    def compute_loss(sample):
        model_y.train()
        if not args.ssl:
            inputs = cast(sample[0], args.dtype)
            targets = cast(sample[1], 'long')
            y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
            if args.dataset == "awa2":
                return F.binary_cross_entropy_with_logits(y, targets.float()), y
            else:
                return F.cross_entropy(y, targets), y
        else:
            global counter

            l = sample[0]
            u = sample[1]
            inputs_l = cast(l[0], args.dtype)
            targets_l = cast(l[1], 'long')
            inputs_u = cast(u[0], args.dtype)

            y_l = data_parallel(model, inputs_l, params, sample[2], list(range(args.ngpu))).float()
            y_u = data_parallel(model, inputs_u, params, sample[2], list(range(args.ngpu))).float()

            if args.dataset == "awa2":
                loss = F.binary_cross_entropy_with_logits(y_l, targets_l.float())
            else:
                loss = F.cross_entropy(y_l, targets_l)

            if args.min_entropy:
                if args.dataset == "awa2":
                    labels_pred = F.sigmoid(y_u)
                    entropy = -torch.sum(labels_pred * torch.log(labels_pred), dim=1)
                else:
                    labels_pred = F.softmax(y_u, dim=1)
                    entropy = -torch.sum(labels_pred * torch.log(labels_pred), dim=1)
                if counter >= 10:
                    loss_entropy = args.unl_weight * torch.mean(entropy)
                    loss += loss_entropy

            elif args.semantic_loss:
                if args.dataset == "awa2":
                    labels_pred = F.sigmoid(y_u)
                else:
                    labels_pred = F.softmax(y_u, dim=1)
                part1 = torch.stack([labels_pred ** all_labels[i] for i in range(all_labels.shape[0])])
                part2 = torch.stack([(1 - labels_pred) ** (1 - all_labels[i]) for i in range(all_labels.shape[0])])
                sem_loss = -torch.log(torch.sum(torch.prod(part1 * part2, dim=2), dim=0))
                if counter >= 10:
                    semantic_loss = args.unl_weight * torch.mean(sem_loss)
                    loss += semantic_loss

            elif args.lp:
                weight = np.min([1., 0.005*counter])
                # weight = 1.

                y_l_full, mu_l, logvar_l = model_y(y_l)
                var_division = logvar_l.exp() / np.exp(prior_log_var)
                diff = mu_l - mu_prior
                diff_term = diff * diff / np.exp(prior_log_var)
                logvar_division = prior_log_var - logvar_l
                # put KLD together
                kld_l = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - num_classes)

                # kld_l = 0.5 * (1 + (inv_sigma1*(logvar_l.exp()) + inv_sigma1*(mu_l.pow(2)) - logvar_l).sum(dim=1) + log_det_sigma)
                # kld_l = 1/2*((log_sigma - logvar_l) + (logvar_l.exp() + mu_l.pow(2))/sigma_prior - 1).sum(dim=1)
                # targets = one_hot_embedding(targets_l, num_classes, device=device)
                recon_loss = F.cross_entropy(y_l_full, targets_l)
                loss = recon_loss.mean()

                # import pdb
                # pdb.set_trace()

                # log_p_theta = torch.logsumexp(log_p_theta_, dim=1) - np.log(x.size(1))
                # print("log_p_theta", log_p_theta.size())
                # log_p_theta = l_p_theta[np.arange(len(targets_l)), targets_l]
                # kld_l = l_q_phi - log_p_theta

                #kld_loss = weight*kld_l.mean()
                #loss += kld_loss
                # import pdb
                # pdb.set_trace()
                
                # import pdb
                # pdb.set_trace()

                if counter >= 25:
                    y_u_full, mu_u, logvar_u = model_y(y_u)

                    var_division = logvar_u.exp() / np.exp(prior_log_var)
                    diff = mu_u - mu_prior
                    diff_term = diff * diff / np.exp(prior_log_var)
                    logvar_division = prior_log_var - logvar_u

                    kld_u = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - num_classes)
                    # kld_u = 0.5 * ((inv_sigma1 * logvar_u.exp() + inv_sigma1 * mu_u.pow(2) - 1 - logvar_u).sum(dim=1) + log_det_sigma)
                    y_u_pred = torch.log_softmax(y_u_full, dim=1)

                    u_loss = ((y_u_pred.exp() * (-y_u_pred)).sum(dim=-1)).mean() # + weight*kld_u.mean()
                    # cross_ent = -(y_u_pred.exp()*y_u_pred).sum(dim=-1)

                    loss += args.unl2_weight * u_loss
                    #loss += args.unl2_weight * (args.unl_weight * weight * kld_u.mean() + cross_ent.mean())

                return loss, y_l_full

            return loss, y_l

    def compute_loss_test(sample):
        model_y.eval()
        inputs = cast(sample[0], args.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
        if args.lp:
            y_full, mu, logvar = model_y(y)
            # kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
            # recon = F.cross_entropy(y_full, targets)
            # tgts = one_hot_embedding(targets, num_classes, device=device)
            recon_loss = F.cross_entropy(y_full, targets)
            return recon_loss.mean(), y_full

        if args.dataset == "awa2":
            return F.binary_cross_entropy_with_logits(y, targets.float()), y
        else:
            return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params=params, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
                   os.path.join(args.save, 'model.pt7'))
        z = {**vars(args), **t}
        with open(os.path.join(args.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        if args.dataset == "awa2":
            if not args.ssl or not state['train']:
                acc = calculate_accuracy(F.sigmoid(state['output'].data), state['sample'][1])
            else:
                acc = calculate_accuracy(F.sigmoid(state['output'].data), state['sample'][0][1])
            classacc.add(acc)
        else:
            if not args.ssl or not state['train']:
                classacc.add(state['output'].data, state['sample'][1])
            else:
                classacc.add(state['output'].data, state['sample'][0][1])
        meter_loss.add(loss)

        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(args, lr * args.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()[0]
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc,
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
              (args.save, state['epoch'], args.epochs, test_acc))

        global counter
        counter += 1

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(compute_loss, train_loader, args.epochs, optimizer)


if __name__ == '__main__':
    main()
