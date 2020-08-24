"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks

    2016 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchnet as tnt
import torch.utils.data as data
from torchnet.engine import Engine

from pathlib import Path
import torch
from torchvision import transforms, datasets

from flows import Invertible1x1Conv, NormalizingFlowModel
from spline_flows import NSF_CL
from torch.distributions import MultivariateNormal
import itertools
from utils import cast, data_parallel, print_tensor_dict
from torch.backends import cudnn
from resnet import resnet
from torch.distributions.dirichlet import Dirichlet
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from logic import (
    cifar10_logic,
    cifar100_logic,
    superclass_mapping,
    super_class_label,
    fc_mapping,
    LogicNet,
    Decoder,
    simplex_transform,
    inverse_simplex_transform,
)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--num_labelled', default=4000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n_workers', default=4, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[10, 50, 100, 150, 200, 250]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)

# Device options
parser.add_argument('--cuda', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate cuda.")
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--sloss", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate sloss.")
parser.add_argument("--unl_weight", type=float, default=0.1,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--sloss_weight", type=float, default=1,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--starter_counter", default=-1, type=int)
# parser.add_argument("--starter_counter", default=10, type=int)


class Joint(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)


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


def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


def convert_to_one_hot(num_categories, labels, device):
    labels = torch.unsqueeze(labels, 1)
    one_hot = torch.FloatTensor(len(labels), num_categories).zero_().to(device)
    one_hot.scatter_(1, labels, 1)
    return one_hot


def mean(numbers):
    if len(numbers) == 0:
        return float(0)
    return float(sum(numbers)) / max(len(numbers), 1)


def resample(hidden_sample, hidden_dim=50):

    mu, logvar = hidden_sample[:, :hidden_dim], hidden_sample[:, hidden_dim:]
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps*std, mu, logvar


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        transformations = [transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []
    transformations.extend([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose(transformations)

    train_dataset = datasets.CIFAR10(dataroot, train=True,
                                     transform=train_transform,
                                     download=download)

    test_dataset = datasets.CIFAR10(dataroot, train=False,
                                    transform=test_transform,
                                    download=download)
    return image_shape, num_classes, train_dataset, test_dataset


def main():
    # device = "cpu"

    opt = parser.parse_args()
    device = "cuda:0" if opt.cuda else "cpu"
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)

    num_classes = 10 if opt.dataset == 'CIFAR10' else 100
    hidden_dim = num_classes

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    global counter, super_class_accuracy, logic_accuracy
    logic_accuracy = []
    super_class_accuracy = []
    counter = 0

    image_shape, num_classes, train_dataset, test_dataset = get_CIFAR10(True, opt.dataroot, True)

    def _init_fn(*args, **kwargs):
        np.random.seed(opt.seed)

    num_labelled = opt.num_labelled
    num_unlabelled = len(train_dataset) - num_labelled

    td_targets = train_dataset.targets
    labelled_idxs, unlabelled_idxs = x_u_split(td_targets, num_labelled, num_classes)
    labelled_set, unlabelled_set = [Subset(train_dataset, labelled_idxs),
                                    Subset(train_dataset, unlabelled_idxs)]

    labelled_set = data.ConcatDataset([labelled_set for i in range(num_unlabelled // num_labelled + 1)])
    labelled_set, _ = data.random_split(labelled_set, [num_unlabelled, len(labelled_set) - num_unlabelled])

    train_dataset = Joint(labelled_set, unlabelled_set)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_workers,
        worker_init_fn=_init_fn
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_workers,
        worker_init_fn=_init_fn
    )

    if opt.dataset == "CIFAR10":
        logic = cifar10_logic
    else:
        logic = cifar100_logic
    global classes
    global sc_mapping, superclass_indexes

    constraint_accuracy, super_class_accuracy = [], []
    superclass_indexes = {}

    f, params = resnet(opt.depth, opt.width, hidden_dim*2)

    if opt.sloss:
        # don't model on the simplex
        # num_flow_classes = num_classes-1
        #
        # prior_y = MultivariateNormal(torch.zeros(num_flow_classes).to(device),
        #                              torch.eye(num_flow_classes).to(device))
        # num_flows = 5
        #
        # flows = [NSF_CL(dim=num_flow_classes, K=8, B=3, hidden_dim=16) for _ in range(num_flows)]
        # convs = [Invertible1x1Conv(dim=num_flow_classes, device=device) for i in range(num_flows)]
        # flows = list(itertools.chain(*zip(convs, flows)))
        #
        # model_y = NormalizingFlowModel(prior_y, flows, num_flow_classes, device=device).to(device)
        # optimizer_y = Adam(model_y.parameters(), lr=1e-3, weight_decay=1e-5)

        logic_net = LogicNet(num_dim=num_classes)
        logic_net.to(device)
        optimizer_logic = Adam(logic_net.parameters(), lr=1e-1, weight_decay=1e-5)
        scheduler = StepLR(optimizer_logic, step_size=50, gamma=0.2)

    decoder = Decoder(hidden_dim=hidden_dim, num_dim=num_classes)
    decoder.to(device)
        # optimizer_decoder = Adam(decoder.parameters(), lr=1e-1, weight_decay=1e-5)
        # scheduler_decoder = StepLR(optimizer_decoder, step_size=50, gamma=0.2)

    def create_optimizer(args, lr):
        print('creating optimizer with lr = ', lr)
        params_ = [v for v in params.values() if v.requires_grad]
        params_ += list(decoder.parameters())
        return SGD(params_, lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        global sc_mapping, counter

        l = sample[0]
        u = sample[1]
        inputs_l = cast(l[0], opt.dtype)
        targets_l = cast(l[1], 'long')
        inputs_u = cast(u[0], opt.dtype)

        hidden_params = data_parallel(f, inputs_l, params, sample[2], list(range(opt.ngpu))).float()
        z, mu, logvar = resample(hidden_params, hidden_dim=hidden_dim)
        y_l = decoder(z)

        loss = F.cross_entropy(y_l, targets_l)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        #
        # if opt.sloss:
        #     # train logic net
        #     samps = torch.softmax(y_l.detach(), dim=1)
        #
        #     logic_net.train()
        #
        #     pred = logic_net(samps)
        #     true = logic(samps).unsqueeze(1).float()
        #
        #     idxs = (true.squeeze(1) == 0)
        #     target_loss = F.binary_cross_entropy(pred, true, reduction="none")
        #
        #     # balance the number of even and non-even samples here??
        #     loss_logic = (target_loss[idxs].sum() + target_loss[~idxs].sum())
        #
        #     optimizer_logic.zero_grad()
        #     loss_logic.backward()
        #     optimizer_logic.step()
        #
        #     if counter > 10:
        #         logic_net.eval()
        #         y_u = data_parallel(f, inputs_u, params, sample[2], list(range(opt.ngpu))).float()
        #         samps = torch.softmax(y_u, dim=1)
        #
        #         pred = logic_net(samps)
        #         true = logic(samps).unsqueeze(1).float()
        #
        #         true_label = torch.ones_like(pred)
        #         idxs = (true.squeeze(1) == 0)
        #
        #         if idxs.sum() > 1:
        #             target_loss = F.binary_cross_entropy(pred, true_label, reduction="none")[idxs]
        #             loss += opt.unl_weight*target_loss.mean()

        return loss + 0.01*kld.mean(), y_l


        # if opt.dataset == "CIFAR100":
        #
        #     sc_log_pred, fc_log_pred, all_pred = log_predictions
        #
        #     true_super_class_label = torch.tensor([super_class_label[superclass_mapping[classes[t]]]
        #                                            for t in targets]).to(device)
        #     true_fine_class_label = torch.tensor([fc_mapping[classes[t]]
        #                                            for t in targets]).to(device)
        #
        #     # fc_pred_labels = fc_log_pred[np.arange(len(true_super_class_label)), true_super_class_label, :]
        #
        #     sc_nll = F.nll_loss(sc_log_pred, true_super_class_label)
        #     fc_nll = F.nll_loss(fc_log_pred, true_fine_class_label)
        #
        #     return sc_nll + fc_nll, all_pred[:, sc_mapping]

    def compute_loss_test(sample):

        global counter, logic_accuracy
        global classes, super_class_accuracy
        global sc_mapping, superclass_indexes

        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        hidden_params = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        z, mu, logvar = resample(hidden_params, hidden_dim=hidden_dim)
        y = decoder(z)

        return F.cross_entropy(y, targets), y

        # log_predictions = logic(y)
        #
        # if opt.dataset == "CIFAR100":
        #
        #     sc_log_pred, fc_log_pred, all_pred = log_predictions
        #
        #     true_super_class_label = torch.tensor(
        #         [super_class_label[superclass_mapping[classes[t]]] for t in targets]).to(device)
        #
        #     super_class_accuracy += list(torch.argmax(sc_log_pred, dim=1) == true_super_class_label)
        #
        #     return F.nll_loss(all_pred[:, sc_mapping], targets), all_pred[:, sc_mapping]
        #
        # return F.nll_loss(log_predictions, targets), log_predictions

    def log(t, state):
        torch.save(dict(params=params, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
                   os.path.join(opt.save, 'model.pt7'))
        z = {**vars(opt), **t}
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        if not state['train']:
            classacc.add(state['output'].data, state['sample'][1])
        else:
            classacc.add(state['output'].data, state['sample'][0][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):

        # with torch.no_grad():
        #     engine.test(compute_loss_test, test_loader)

        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        global constraint_accuracy, super_class_accuracy, logic_accuracy

        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()

        classacc.reset()
        timer_test.reset()
        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]

        constraint_accuracy_val = mean(logic_accuracy)
        logic_accuracy = []

        if opt.sloss:
            scheduler.step()

        # sch_enc.step()
        # sch_dec.step()
        # sch_log.step()
        # super_class_accuracy_val = mean(super_class_accuracy)
        # super_class_accuracy = []

        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "logic_accuracy": constraint_accuracy_val
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m, const_acc: \33[91m%.2f\033[0m' %
              (opt.save, state['epoch'], opt.epochs, test_acc, constraint_accuracy_val))

        global counter
        counter += 1

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
