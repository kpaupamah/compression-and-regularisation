import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

from quantiser import QuantizableMobileNetV2, QuantizableShuffleNetV2


def load_data(args, use_cuda, valid_size=0):
    train_data, test_data = None, None

    if args.data == "cifar":
        print("=" * 60)
        print("LOADING CIFAR10")
        print("=" * 60)

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    elif args.data == "fmnist":
        print("=" * 60)
        print("LOADING FashionMNIST")
        print("=" * 60)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        if args.arch == "alexnet":
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])

        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, val_loader, train_data, test_data


def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        # model.reset_parameters()
        model.eval()
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()


def deactivate_dropout(model):
    if isinstance(model, nn.Dropout):
        model.eval()


def get_model(args, n_classes=10):
    network = None

    if args.arch == "alexnet":
        print("=" * 60)
        print("ALEXNET")
        print("=" * 60)

        network = models.alexnet()

        if args.data == "fmnist":
            network.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

        n_inputs = network.classifier[6].in_features
        network.classifier[6] = nn.Linear(in_features=n_inputs, out_features=n_classes, bias=True)

        if args.overfit == 1:
            network.apply(deactivate_dropout)

        if args.train == 0 and args.qload:
            print("loading quantised model: " + args.qload)
            network = torch.jit.load('models/' + args.data + '/' + args.qload)
        elif args.load:
            print("loading model: " + args.load)
            network.load_state_dict(torch.load('models/' + args.data + '/' + args.load))


    elif args.arch == "mobilenet":
        print("=" * 60)
        print("MOBILENET")
        print("=" * 60)

        if args.quantise == 1:
            network = QuantizableMobileNetV2()
        else:
            network = models.mobilenet_v2()

        if args.data == "fmnist":
            network.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        n_inputs = network.classifier[1].in_features
        network.classifier[1] = nn.Linear(in_features=n_inputs, out_features=n_classes)

        if args.overfit == 1:
            network.apply(deactivate_batchnorm)
            network.apply(deactivate_dropout)

        if args.train == 0 and args.qload:
            print("loading quantised model: " + args.qload)
            network = torch.jit.load('models/' + args.data + '/' + args.qload)
        if args.load:
            print("loading model: " + args.load)
            network.load_state_dict(torch.load('models/' + args.data + '/' + args.load))


    elif args.arch == "shufflenet":
        print("=" * 60)
        print("SHUFFLENET")
        print("=" * 60)

        if args.quantise == 1:
            network = QuantizableShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024])
        else:
            network = models.shufflenet_v2_x1_0()

        if args.data == "fmnist":
            network.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        n_inputs = network.fc.in_features
        network.fc = nn.Linear(in_features=n_inputs, out_features=n_classes)

        if args.overfit == 1:
            network.apply(deactivate_batchnorm)

        if args.train == 0 and args.qload:
            print("loading quantised model: " + args.qload)
            network = torch.jit.load('models/' + args.data + '/' + args.qload)
        if args.load:
            print("loading model: " + args.load)
            network.load_state_dict(torch.load('models/' + args.data + '/' + args.load))

    return network



def get_num_weights(model, verbose=False):
    if verbose:
        print("\n" + model.__class__.__name__ + "\n")
    parameters = []
    total = 0
    for param_name, param in model.named_parameters():
        if 'weight' in param_name:
            if verbose:
                print(param_name, param.nonzero().size(0))
            parameters.append([param_name, param.nonzero().size(0)])
            total += param.nonzero().size(0)

    if verbose:
        print("\nTotal:", total)

    return parameters, total


def get_rules(path):
    file = open(path, "r")
    contents = file.readlines()
    file.close()

    rule = []
    for i, line in enumerate(contents):
        p_name, sparsity, importance, _ = line.split(",")
        sparsity = sparsity.split(";")
        rule.append([p_name, [float(sparsity[0]), float(sparsity[1])], importance])

    return rule


def set_sparsity(model, sparsity, name=None):

    if not name:
        name = model.__class__.__name__
    file = open("rules/" + name + ".rule", "w")
    for i, (param_name, param) in enumerate(model.named_parameters()):
        if 'weight' in param_name:
            if 'fc' in param_name or 'classifier' in param_name:
                file.write(param_name + "," + str(sparsity) + ";" + str(sparsity) + "," + "abs" + ",\n")
            else:
                file.write(param_name + "," + str(sparsity) + ";" + str(sparsity) + "," + "l2norm" + ",\n")

    file.close()


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
