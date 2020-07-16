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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from flows import Invertible1x1Conv, NormalizingFlowModel
from spline_flows import NSF_CL
from torch.distributions import MultivariateNormal
import itertools
from utils import cast, data_parallel, print_tensor_dict
from torch.backends import cudnn
from resnet import resnet
from torch.distributions.dirichlet import Dirichlet


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



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
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
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


def main():
    # device = "cpu"

    opt = parser.parse_args()
    device = "cuda:0" if opt.cuda else "cpu"
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    global counter
    counter = 0

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    global classes
    global superclass_labels
    global superclass_indexes
    global constraint_accuracy

    constraint_accuracy = []

    classes = train_loader.dataset.classes
    superclass_labels = [super_class_label[superclass_mapping[c]] for c in classes]
    superclass_indexes = {}
    all_labels = torch.eye(len(super_class_label)).to(device)

    for cat in range(len(super_class_label)):
        indices = [i for i, x in enumerate(superclass_labels) if x == cat]
        superclass_indexes[cat] = indices

    f, params = resnet(opt.depth, opt.width, num_classes)

    if opt.sloss:
        num_flow_classes = num_classes
        prior_y = MultivariateNormal(torch.zeros(num_flow_classes).to(device),
                                     torch.eye(num_flow_classes).to(device))
        num_flows = 3
        flows = [NSF_CL(dim=num_flow_classes, K=8, B=3, hidden_dim=16, device=device) for _ in range(num_flows)]
        convs = [Invertible1x1Conv(dim=num_flow_classes, device=device) for i in range(num_flows)]
        flows = list(itertools.chain(*zip(convs, flows)))
        model_flow = NormalizingFlowModel(prior_y, flows, num_flow_classes, device=device).to(device)
        optimizer_flow = Adam(model_flow.parameters(), lr=1e-3, weight_decay=1e-5)

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay)

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
        global counter
        global classes
        global superclass_labels, superclass_indexes

        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        loss_prediction = F.cross_entropy(y, targets)

        # add the normalizing flows logic layer here
        if opt.sloss:

            if counter >= 10:

                model_flow.eval()

                labels_pred = F.softmax(y, dim=1)
                # calc likelihood of this prediction under constraints
                _, nll_ypred = model_flow(labels_pred)

                loss_nll_ypred = opt.unl_weight * torch.mean(nll_ypred)
                loss_prediction += loss_nll_ypred


        # train the flow to follow the logical specification
        model_flow.train()
        optimizer_flow.zero_grad()

        # train on true samples
        one_hot_targets = F.one_hot(torch.tensor(targets), num_classes).float()
        one_hot_targets = one_hot_targets * 120 + (1 - one_hot_targets) * 1.1
        dirichlet_targets = torch.stack([Dirichlet(i).sample() for i in one_hot_targets])
        zs, nll_y = model_flow(dirichlet_targets)
        loss_nll_y = torch.mean(nll_y)
        loss_flow = loss_nll_y

        if counter >= 10:
            # train on generated samples
            prior_sample = model_flow.prior.sample((one_hot_targets.size(0),))
            xs, log_det_back = model_flow.backward(prior_sample)
            predictions = F.log_softmax(xs[-1], dim=1)
            # true_super_class_label = torch.tensor([super_class_label[superclass_mapping[classes[t]]] for t in targets])
            superclass_predictions = torch.cat([predictions[:, superclass_indexes[c]].logsumexp(dim=1).unsqueeze(1)
                                                for c in range(len(super_class_label))], dim=1).exp()

            part1 = torch.stack([superclass_predictions ** all_labels[i] for i in range(all_labels.shape[0])])
            part2 = torch.stack([(1 - superclass_predictions) ** (1 - all_labels[i]) for i in range(all_labels.shape[0])])

            sloss = -torch.log(torch.sum(torch.prod(part1 * part2, dim=2), dim=0))
            loss_bkwd = -(torch.mean(sloss) + log_det_back.mean())

            loss_flow += loss_bkwd

        loss_flow.backward()
        optimizer_flow.step()

        return loss_prediction, y

    def compute_loss_test(sample):

        global counter
        global classes
        global superclass_labels, superclass_indexes
        global constraint_accuracy

        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        loss_prediction = F.cross_entropy(y, targets)

        predictions = F.log_softmax(y, dim=1)
        superclass_predictions = torch.cat([predictions[:, superclass_indexes[c]].logsumexp(dim=1).unsqueeze(1)
                                            for c in range(len(super_class_label))], dim=1).exp()
        true_super_class_label = torch.tensor([super_class_label[superclass_mapping[classes[t]]] for t in targets])
        constraint_accuracy += list(torch.argmax(superclass_predictions, dim=1) == true_super_class_label)

        return loss_prediction, y

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
        classacc.add(state['output'].data, state['sample'][1])
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
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        global constraint_accuracy

        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()

        classacc.reset()
        timer_test.reset()
        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]

        def mean(numbers):
            return float(sum(numbers)) / max(len(numbers), 1)

        constraint_accuracy_val = mean(constraint_accuracy)
        constraint_accuracy = []

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
            "constraint_acc": constraint_accuracy_val
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
              (opt.save, state['epoch'], opt.epochs, test_acc))

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