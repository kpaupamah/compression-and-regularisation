import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

import numpy as np
import random
import time
import argparse
import os


from utils import load_data, get_rules, get_model, get_num_weights, set_sparsity, weights_init_uniform_rule
from trainer import run_training, test
from pruner import Pruner
from quantiser import quantise

# from experiments import run_sparsity_scan, run_sensitivity_scan, run_overfitting, iter_prune, retrain, overfit_model, get_max_sparsity
# from prune import apply_weight_sharing

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--val_size', type=float, default=0.2)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--data', type=str, default="cifar")
parser.add_argument('--arch', type=str, default="alexnet")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--decay_lr', type=int, default=0)
parser.add_argument('--lr_decay_step', type=int, default=5)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--prune', type=int, default=0)
parser.add_argument('--prune_iter', type=int, default=5)
parser.add_argument('--sensitivity', type=float, default=0.3)
parser.add_argument('--init_param', type=int, default=0)
parser.add_argument('--overfit', type=int, default=0)
parser.add_argument('--qload', type=str, default=None)
parser.add_argument('--quantise', type=int, default=0)
parser.add_argument('--use_cuda', type=int, default=1)

args = parser.parse_args()

assert not (args.prune == 1 and not args.load), "Specify a model to prune with --load"
assert not (args.overfit == 1 and not args.load), "Specify a model to overfit with --load"
assert not (args.quantise == 1 and not args.load), "Specify a model to quantise with --load"
assert not ((args.quantise == 1 or args.qload) and args.use_cuda == 1), "Quantised models do not support CUDA!"
assert not (args.quantise == 1 and args.qload), "Model already quantised; set --quantise=0 and use --test=1 with --qload instead to test accuracy."
assert not (args.quantise == 1 and args.train == 1), "Cannot train a quantised model."
assert not (args.qload and (args.train == 1 or args.overfit == 1)), "Cannot train a quantised model."
assert not (args.qload and args.prune == 1), "Cannot prune a quantised model."

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

os.environ['TORCH_HOME'] = 'models'

use_cuda = torch.cuda.is_available() and args.use_cuda
device = torch.device("cuda" if use_cuda else "cpu")
print("\nUSING", "CUDA" if use_cuda else "CPU")
print("")

# load data
print("Loading data...")
train_loader, test_loader, val_loader, train_data, test_data = load_data(args, use_cuda, valid_size=args.val_size)
n_classes = len(test_data.classes)

# initialise network
print("Initialising network...")
network = get_model(args)

# map classes to indexes
network.class_to_idx = train_data.class_to_idx
network.idx_to_class = {
    idx: class_
    for class_, idx in network.class_to_idx.items()
}

# print(network)
# get_num_weights(network, verbose=True)


network = network.to(device)

optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()


fname = network.__class__.__name__
if args.fname:
    fname = args.fname


if args.train == 1:

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print("")
    run_training(args, network, device, train_loader, test_loader, optimizer, criterion, fname=fname)


if args.test == 1:
    print("=" * 60)
    print("TESTING")
    print("=" * 60)
    print("")
    s = time.time()
    test(network, device, test_loader)
    e = time.time()
    print("Inference Time:  " + str(e - s))


if args.prune == 1:
    print("=" * 60)
    print("PRUNING")
    print("=" * 60)
    print("")

    name = args.data + '_' + args.load[:-4]
    set_sparsity(network, args.sensitivity, name)
    rule = get_rules("rules/" + name + ".rule")
    fname = args.load[:-4] + '_pruned'
    original_param, o_total = get_num_weights(network, verbose=False)

    pruner = Pruner(rule=rule)
    pruner.prune(model=network, stage=0, update_masks=True, verbose=False)

    if args.init_param == 1:
        network.apply(weights_init_uniform_rule)
        print("\nRe-initialised weights...")

    # prune
    for i in range(args.prune_iter):
        print("")
        print("-" * 60)
        print("PRUNE ITERATION", i)
        print("-" * 60)
        print("")

        run_training(args, network, device, train_loader, test_loader, optimizer, criterion, fname, pruner=pruner)
        pruner.prune(model=network, stage=0, update_masks=True, verbose=False)

    remainder_param, r_total = get_num_weights(network, verbose=False)
    print("\nWeight Layers:\t\tOriginal Parameters:\t\tRemaining Parameters:\t\tNum Pruned")
    for i in range(len(original_param)):
        name, num_old = original_param[i]
        num_new = remainder_param[i][1]
        print("{:^30}\t\t{}\t\t{}\t\t{}".format(name, num_old, num_new, num_old - num_new))

    print("\nTotal Original: {}\t\tRemaining: {}\tPruned: {}".format(o_total, r_total, o_total - r_total))


if args.quantise == 1:
    print("")
    print("=" * 60)
    print("QUANTISING")
    print("=" * 60)
    s = time.time()
    network = quantise(args, network, device, val_loader)
    e = time.time()
    print("Quantisation Time: " + str(e - s))

    print("\nTesting quantisation...")
    s = time.time()
    test(network, device, test_loader)
    e = time.time()
    print("Inference Time:  " + str(e - s))



if args.overfit == 1:
    print("=" * 60)
    print("OVERFITTING")
    print("=" * 60)
    print("")

    fname = args.load[:-4] + '_overfitted'
    run_training(args, network, device, train_loader, test_loader, optimizer, criterion, fname=fname, overfit=True,
                 val_loader=val_loader)