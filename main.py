import argparse
import os
import json
from tqdm import tqdm

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
from torch.utils.data import Subset
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_

import torchnet as tnt
from torchnet.engine import Engine
from utils import *
from torch.backends import cudnn
from resnet import resnet
from datasets import Joint, check_dataset

from logic import (
    DecoderModel,
    cifar100_logic,
    LogicNet,
    set_class_mapping,
    get_cifar100_pred,
    get_true_cifar100_sc,
    get_cifar100_unnormed_pred,
    get_true_cifar100_from_one_hot
)


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
parser.add_argument("--sloss_weight", type=float, default=1.,
                    help="Weight for unlabelled regularizer loss")
parser.add_argument("--num_hidden", type=int, default=10,
                    help="Dim of the latent dimension used")


def idx_to_one_hot(labels, num_classes, device):
    y_onehot = torch.FloatTensor(len(labels), num_classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.unsqueeze(1), 1)
    return y_onehot


def warmup_decoder_model(net, logic_net, opt, logic_opt, scheduler, logic_scheduler, num_iter=2500, device="cpu"):

    losses = []
    logic_losses = []

    for i in tqdm(range(num_iter)):

        net.eval()
        logic_net.train()
        targets = torch.arange(10).repeat(100)
        tgt = idx_to_one_hot(targets, 10, device)

        # train logic to recognise truth
        pred = logic_net(tgt)
        true = torch.ones_like(pred)
        logic_loss = F.binary_cross_entropy_with_logits(pred, true)

        # train logic loss to recognise true logic
        samples = torch.softmax(net.sample(1000), dim=1)
        pred = logic_net(samples).squeeze(1)
        true = (samples > 0.95).any(dim=1).float()

        logic_loss += F.binary_cross_entropy_with_logits(pred, true)

        logic_opt.zero_grad()
        logic_loss.backward()
        clip_grad_norm_(logic_loss, 1)
        logic_opt.step()

        logic_net.eval()
        net.train()

        logic_losses.append(logic_loss.item())

        weight = np.min([1, i / 1000])
        # train net to reconstruct input
        recon, (z, mu, logvar) = net(tgt)
        recon_loss = F.cross_entropy(recon, targets)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean())

        # train the net to follow the logic
        samples = torch.softmax(net.sample(1000), dim=1)
        pred = logic_net(samples).squeeze(1)
        true = (samples > 0.95).any(dim=1)

        loss_ = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction="none")
        loss = recon_loss + weight * KLD

        if i > 100:
            loss += 0.1 * weight * loss_[~true].sum() / len(loss_)

        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(loss, 1)
        opt.step()

        scheduler.step()
        logic_scheduler.step()

        losses.append((recon_loss + KLD).item())

    targets = torch.arange(10).repeat(1000)
    tgt = idx_to_one_hot(targets, 10, device)
    recon, (z, mu, logvar) = net(tgt)
    print((recon.softmax(dim=1).argmax(dim=1) == targets).detach().numpy().mean())


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()
    print('parsed options:', vars(args))
    epoch_step = json.loads(args.epoch_step)

    check_manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ds = check_dataset(args.dataset, args.dataroot, args.download)
    image_shape, num_classes, train_dataset, test_dataset, td_targets = ds
    num_super_classes = 20

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

    class_names = test_dataset.classes

    if args.generative_loss:
        model_y = DecoderModel(num_classes, z_dim, device)
        model_y.to(device)
        model_y.apply(init_weights)
        model_y.train()
        opt_y = Adam(model_y.parameters(), 1e-2)
        scheduler = StepLR(opt_y, step_size=5, gamma=0.9)

        logic_net = LogicNet(num_classes)
        logic_net.to(device)
        logic_net.apply(init_weights)
        logic_opt = Adam(logic_net.parameters(), lr=1e-2)
        scheduler2 = StepLR(logic_opt, step_size=40, gamma=0.2)

    def create_optimizer(args, lr):
        print('creating optimizer with lr = ', lr)
        params_ = [v for v in params.values() if v.requires_grad]
        if args.generative_loss:
            params_ += model_y.parameters()
        return SGD(params_, lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer(args, args.lr)

    epoch = 0

    # print('\nParameters:')
    # print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    if args.generative_loss:
        n_parameters += sum(p.numel() for p in model_y.get_local_params())
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()

    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    global superclassacc
    superclassacc = []

    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    global counter, classes, logic_losses
    logic_losses = 0
    if args.dataset == "cifar100":
        classes = test_dataset.classes
        set_class_mapping(classes)
    counter = 0

    if args.resume != '':
        state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

        model_y.load_state_dict(state_dict['model_y'])
        # logic_net.load_state_dict(state_dict['logic_net'])

    # warmup and initialise the decoder model
    # warmup_decoder_model(model_y, logic_net, opt_y, logic_opt, scheduler, scheduler2, device=device)

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
            global counter, logic_losses

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

                model_y.eval()
                logic_net.train()
                tgt = idx_to_one_hot(targets_l, 10, device)

                # train logic to recognise truth
                pred = logic_net(tgt)
                true = torch.ones_like(pred)
                logic_loss = F.binary_cross_entropy_with_logits(pred, true)

                # train logic loss to recognise true logic
                samples = torch.softmax(model_y.sample(1000).detach(), dim=1)
                pred = logic_net(samples).squeeze(1)
                true = (samples > 0.95).any(dim=1).float().to(device)

                logic_loss += F.binary_cross_entropy_with_logits(pred, true)

                logic_opt.zero_grad()
                logic_loss.backward()
                clip_grad_norm_(logic_loss, 1)
                logic_opt.step()

                logic_net.eval()
                model_y.train()

                recon, (z, mu, logvar) = model_y(y_l)
                recon_loss = F.cross_entropy(recon, targets_l)
                KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean())

                weight = np.min([1., (counter+1)/100])
                loss = recon_loss + weight*KLD

                if counter > 5:
                    samples = torch.softmax(model_y.sample(len(y_l)), dim=1)
                    pred = logic_net(samples).squeeze(1)
                    true = (samples > 0.95).any(dim=1).to(device)

                    loss_ = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction="none")
                    loss += args.unl2_weight * weight * loss_[~true].sum() / len(loss_)

                if counter > 10:
                    y_u1 = data_parallel(model, inputs_u, params, sample[3], list(range(args.ngpu))).float()
                    recon, (z, mu, logvar) = model_y(y_u1)

                    log_pred = torch.log_softmax(recon, dim=1)

                    entropy = -(log_pred.exp()*log_pred).sum(dim=1).mean()
                    KLD_u = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean())
                    loss += args.unl_weight * (entropy + weight*KLD_u)

                    # pred = log_pred.exp()
                    # logic_u_pred = logic_net(pred).squeeze(1)
                    # logic_u_true = (pred > 0.95).any(dim=1).to(device)
                    #
                    # loss_u = F.binary_cross_entropy_with_logits(logic_u_pred, torch.ones_like(logic_u_pred), reduction="none")
                    # loss += args.unl2_weight * weight * loss_u[~logic_u_true].sum() / len(loss_u)

                return loss, recon

            return loss, y_l

    def compute_loss_test(sample):
        global superclassacc
        model_y.eval()
        inputs = cast(sample[0], args.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
        if args.generative_loss:
            y_full, latent = model_y(y)
            # if args.dataset == "cifar100":
            #     log_pred = torch.log_softmax(y_full, dim=-1)
            #     sc_pred = get_cifar100_pred(log_pred)
            #     sc_targets = get_true_cifar100_sc(targets, classes).to(device)
            #     superclassacc.add(sc_pred, sc_targets)

            recon_loss = F.cross_entropy(y_full, targets)

            # logic
            log_prob = torch.log_softmax(y_full, dim=1)
            true_logic_pred = (log_prob.exp() > 0.95).any(dim=1)

            # logic_in = torch.cat((log_prob, idx_to_one_hot(targets, num_classes, device)), dim=1)
            # pred = logic_net(logic_in).squeeze()

            # true_logic = cifar100_logic(probabilities, torch.argmax(probabilities), class_names).float()
            superclassacc += true_logic_pred.tolist()

            # y_pred = torch.argmax(probabilities, dim=1)
            # import pdb
            # pdb.set_trace()
            #
            # from sklearn.metrics import confusion_matrix
            # import matplotlib.pyplot as plt
            # confusion_matrix(targets, y_pred)
            #
            # # import pdb
            # # pdb.set_trace()
            #
            # arr = ((y_pred == 3) & (targets == 2))
            # for img in inputs[arr]:
            #     img_vec = img.detach().numpy().transpose(1,2,0).astype(np.float) * .5 + .5
            #     plt.imshow(img_vec)
            #     plt.show()
            #
            # import pdb
            # pdb.set_trace()

            return recon_loss.mean(), y_full

        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(
            params=params,
            epoch=t['epoch'],
            optimizer=state['optimizer'].state_dict(),
            model_y=model_y.state_dict(),
            logic_net=logic_net.state_dict(),
            logic_opt=logic_opt.state_dict(),
            # opt_y=opt_y.state_dict()
        ),
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

        # with torch.no_grad():
        #     engine.test(compute_loss_test, test_loader)

    def on_end_epoch(state):
        global superclassacc, logic_losses

        train_loss = meter_loss.value()
        train_acc = classacc.value()[0]
        train_time = timer_train.value()

        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        superclassacc = []

        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]
        sc_acc = np.mean(superclassacc)

        # scheduler.step()
        scheduler2.step()

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
            "logic_acc": sc_acc,
            "train_logic_loss": logic_losses
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % (args.save, state['epoch'], args.epochs, test_acc))

        global counter
        logic_losses = 0
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
