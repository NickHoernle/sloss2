import argparse
import os
import json
from tqdm import tqdm

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
from torch.utils.data import Subset
from torchvision import transforms

import torchnet as tnt
from torchnet.engine import Engine
from utils import *
from torch.backends import cudnn
from resnet import resnet
from datasets import Joint, check_dataset

from logic import DecoderModel, cifar100_logic, LogicNet, set_class_mapping, get_cifar100_pred, get_true_cifar100_sc


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
parser.add_argument('--weight_decay', default=0.0004, type=float)
parser.add_argument('--epoch_step', default='[60, 120, 160]', type=str,
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
parser.add_argument("--semantic_loss", action="store_true",
                    help="Add the semantic loss")
parser.add_argument("--generative_loss", action="store_true",
                    help="Add the generative loss")
parser.add_argument("--unl_weight", type=float, default=0.1,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--unl2_weight", type=float, default=0.1,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--num_hidden", type=int, default=10,
                    help="Dim of the latent dimension used")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()
    print('parsed options:', vars(args))
    epoch_step = json.loads(args.epoch_step)

    check_manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ds = check_dataset(args.dataset, args.dataroot, args.download)
    image_shape, num_classes, train_dataset, test_dataset, td_targets = ds

    if args.ssl:
        num_labelled = args.num_labelled
        num_unlabelled = len(train_dataset) - num_labelled

        labelled_idxs, unlabelled_idxs = x_u_split(td_targets, num_labelled, num_classes)
        labelled_set, unlabelled_set = [Subset(train_dataset, labelled_idxs),
                                            Subset(train_dataset, unlabelled_idxs)]
        labelled_set = data.ConcatDataset([labelled_set for i in range(num_unlabelled // num_labelled + 1)])
        labelled_set, _ = data.random_split(labelled_set, [num_unlabelled, len(labelled_set) - num_unlabelled])

        transformations = [transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        train_dataset = Joint(labelled_set, unlabelled_set, transform=transformations)

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

    z_dim = args.num_hidden
    model, params = resnet(args.depth, args.width, num_classes, image_shape[0])

    if args.generative_loss:
        model_y = DecoderModel(num_classes, z_dim)
        model_y.to(device)
        model_y.apply(init_weights)
        model_y.train()

        if args.dataset == "cifar100":
            logic_net = LogicNet(num_classes)
            logic_net.to(device)
            logic_net.apply(init_weights)
            logic_opt = Adam(logic_net.parameters(), lr=1e-3)

    def create_optimizer(args, lr):
        print('creating optimizer with lr = ', lr)
        params_ = [v for v in params.values() if v.requires_grad]
        if args.generative_loss:
            params_ += model_y.parameters()
        return SGD(params_, lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer(args, args.lr)

    epoch = 0

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    if args.generative_loss:
        n_parameters += sum(p.numel() for p in model_y.parameters())
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()

    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    superclassacc = tnt.meter.ClassErrorMeter(accuracy=True)

    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    global counter
    if args.dataset == "cifar100":
        set_class_mapping(test_dataset.classes)
    counter = 0

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

    def compute_loss(sample):

        if not args.ssl:
            inputs = cast(sample[0], args.dtype)
            targets = cast(sample[1], 'long')
            y = data_parallel(model, inputs, params, sample[3], list(range(args.ngpu))).float()
            if args.dataset == "awa2":
                return F.binary_cross_entropy_with_logits(y, targets.float()), y
            else:
                return F.cross_entropy(y, targets), y
        else:
            global counter

            l = sample[0]
            u1 = sample[1]
            u2 = sample[2]

            inputs_l = cast(l[0], args.dtype)
            targets_l = cast(l[1], 'long')
            inputs_u = cast(u1[0], args.dtype)
            inputs_u2 = cast(u2[0], args.dtype)

            y_l = data_parallel(model, inputs_l, params, sample[3], list(range(args.ngpu))).float()
            loss = F.cross_entropy(y_l, targets_l)

            if args.semantic_loss:
                y_u = data_parallel(model, inputs_u, params, sample[3], list(range(args.ngpu))).float()
                labels_pred = F.softmax(y_u, dim=1)
                all_labels = torch.eye(num_classes).to(device)
                part1 = torch.stack([labels_pred ** all_labels[i] for i in range(all_labels.shape[0])])
                part2 = torch.stack([(1 - labels_pred) ** (1 - all_labels[i]) for i in range(all_labels.shape[0])])
                sem_loss = -torch.log(torch.sum(torch.prod(part1 * part2, dim=2), dim=0))
                if counter >= 10:
                    semantic_loss = args.unl_weight * torch.mean(sem_loss)
                    loss += semantic_loss

            elif args.generative_loss:

                ixs = np.arange(len(y_l))

                if args.dataset == "cifar100":
                    logic_net.train()
                    gen_samples, _ = model_y.train_generative_only(len(targets_l))
                    logic_loss = 0
                    for cat in range(num_classes):
                        log_pred = torch.log_softmax(gen_samples[:, cat, :].detach(), dim=-1)
                        true_logic = cifar100_logic(log_pred)
                        predicted_logic = logic_net(log_pred).squeeze()
                        logic_loss += F.binary_cross_entropy(predicted_logic, true_logic.float())

                    logic_opt.zero_grad()
                    logic_loss.backward()
                    logic_opt.step()
                    logic_net.eval()

                # custom generator loss
                loss = 0
                log_preds_g, latent = model_y.train_generative_only(len(targets_l))
                for cat in range(num_classes):
                    fake_tgts = torch.ones_like(targets_l) * cat
                    log_pred = torch.log_softmax(log_preds_g[:, cat, :], dim=-1)
                    loss += F.nll_loss(log_pred, fake_tgts)

                    if args.dataset == "cifar100":
                        # chances that the samples break the logic
                        true_logic = cifar100_logic(log_pred)
                        predicted_logic = logic_net(log_pred).squeeze()
                        fake = torch.ones_like(predicted_logic)
                        logic_loss2 = F.binary_cross_entropy_with_logits(predicted_logic, fake, reduction="none")
                        logic_loss2 = logic_loss2[~true_logic].sum() / len(predicted_logic)
                        loss += logic_loss2

                log_preds, latent = model_y(y_l)
                (z, mu, logvar, cmu_, clv_) = latent

                loss += F.cross_entropy(log_preds, targets_l)

                # encoder loss
                cmu = cmu_[ixs, targets_l]
                clv = clv_[ixs, targets_l]

                nll = args.unl2_weight*(-log_normal(z, cmu, clv) + log_normal(z, mu, logvar)).mean()
                loss += nll

                # unsupervised part
                if counter > 20:
                    y_u = data_parallel(model, inputs_u, params, sample[3], list(range(args.ngpu))).float()
                    y_u2 = data_parallel(model, inputs_u2, params, sample[3], list(range(args.ngpu))).float()

                    log_preds_u, latent_u = model_y(y_u)
                    log_preds_u2, latent_u2 = model_y(y_u2)

                    (z, mu, logvar, cluster_mus, cluster_logvars) = latent_u
                    log_predictions2 = torch.log_softmax(log_preds_u2, dim=1)
                    z_expanded = z.unsqueeze(1).repeat(1, num_classes, 1)
                    reconstruction = (-(log_predictions2.exp()*log_normal(z_expanded, cluster_mus, cluster_logvars)).sum(dim=1) + log_normal(z, mu, logvar)).mean()

                    (z2, mu2, logvar2, cluster_mus2, cluster_logvars2) = latent_u2
                    log_predictions = torch.log_softmax(log_preds_u, dim=1)
                    z_expanded2 = z2.unsqueeze(1).repeat(1, num_classes, 1)
                    reconstruction2 = (-(log_predictions.exp() * log_normal(z_expanded2, cluster_mus2, cluster_logvars2)).sum(dim=1) + log_normal(z2, mu2, logvar2)).mean()

                    loss += args.unl_weight*(reconstruction+reconstruction2)

                return loss, log_preds

            return loss, y_l

    def compute_loss_test(sample):
        model_y.eval()
        inputs = cast(sample[0], args.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
        if args.generative_loss:
            y_full, latent = model_y(y)
            if args.dataset == "cifar100":
                log_pred = torch.log_softmax(y_full, dim=-1)
                superclassacc.add(get_cifar100_pred(log_pred), get_true_cifar100_sc(targets).to(device))

            recon_loss = F.cross_entropy(y_full, targets)
            return recon_loss.mean(), y_full

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
        superclassacc.reset()

        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]
        sc_acc = superclassacc.value()[0]

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
            "super_class_acc": sc_acc,
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % (args.save, state['epoch'], args.epochs, test_acc))

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
