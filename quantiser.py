import os
import torch
import torch.nn as nn
import torchvision.models as models


def quantise(args, model, device, dataset_loader):
    print("Preparing quantisation...")
    model.eval()
    if args.arch == "alexnet":
        fuse_model_alexnet(model)
    elif args.arch == "mobilenet":
        fuse_model_mobilenet(model)
    elif args.arch == "shufflenet":
        fuse_model_shufflenet(model)

    model = torch.quantization.QuantWrapper(model)
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    print("Calibrating...")
    calibrate(model, device, dataset_loader)

    print("Quantizing...")
    torch.quantization.convert(model, inplace=True)


    original_fpath = 'models/' + args.data + '/' + args.load
    fpath = 'models/' + args.data + '/' + args.load[:-4] + '_quantised.pth'
    torch.jit.save(torch.jit.script(model), fpath)

    fsize = os.path.getsize(original_fpath) / 1024 / 1024
    qfsize = os.path.getsize(fpath) / 1024 / 1024
    print("Quantised filesize: %.4fMB     (original: %.4fMB)" % (qfsize, fsize))

    return model


def calibrate(model, device, loader):
    model.eval()
    with torch.no_grad():
        progress = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            progress += len(data)
            # print(progress,'/',len(loader.sampler))
            break


def fuse_model_shufflenet(network):
    # ShuffleNetV2
    network.conv1[2] = nn.ReLU(inplace=False)
    network.conv1[2].eval()
    torch.quantization.fuse_modules(network.conv1, ['0', '1', '2'], inplace=True)
    network.conv5[2] = nn.ReLU(inplace=False)
    network.conv5[2].eval()
    torch.quantization.fuse_modules(network.conv5, ['0', '1', '2'], inplace=True)
    for m in network.modules():
        if hasattr(m, 'branch2'):
            b2 = m.branch2
            b2[2] = nn.ReLU(inplace=False)
            b2[7] = nn.ReLU(inplace=False)
            b2[2].eval();
            b2[7].eval();
            torch.quantization.fuse_modules(b2, ['0', '1', '2'], inplace=True)
            torch.quantization.fuse_modules(b2, ['3', '4'], inplace=True)
            torch.quantization.fuse_modules(b2, ['5', '6', '7'], inplace=True)
        if hasattr(m, 'branch1'):
            b1 = m.branch1
            if isinstance(b1, torch.nn.Sequential):
                torch.quantization.fuse_modules(b1, ['0', '1'], inplace=True)
                b1[4] = nn.ReLU(inplace=False);
                b1[4].eval()
                torch.quantization.fuse_modules(b1, ['2', '3', '4'], inplace=True)


def fuse_model_mobilenet(network):
    # MobileNetV2
    for m in network.modules():
        if isinstance(m, models.mobilenet.ConvBNReLU):
            m[2] = nn.ReLU(inplace=False)
            m[2].eval()
            torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        if isinstance(m, models.mobilenet.InvertedResidual) or isinstance(m, MobilenetInvertedResidual):
            for idx in range(len(m.conv)):
                if type(m.conv[idx]) == torch.nn.Conv2d:
                    torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def fuse_model_alexnet(network):
    # AlexNet
    network.features[1] = nn.ReLU(inplace=False)
    network.features[4] = nn.ReLU(inplace=False)
    network.features[7] = nn.ReLU(inplace=False)
    network.features[9] = nn.ReLU(inplace=False)
    network.features[11] = nn.ReLU(inplace=False)
    network.classifier[2] = nn.ReLU(inplace=False)
    network.classifier[5] = nn.ReLU(inplace=False)
    network.eval()
    torch.quantization.fuse_modules(network.features, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(network.features, ['3', '4'], inplace=True)
    torch.quantization.fuse_modules(network.features, ['6', '7'], inplace=True)
    torch.quantization.fuse_modules(network.features, ['8', '9'], inplace=True)
    torch.quantization.fuse_modules(network.features, ['10', '11'], inplace=True)
    torch.quantization.fuse_modules(network.classifier, ['1', '2'], inplace=True)
    torch.quantization.fuse_modules(network.classifier, ['4', '5'], inplace=True)



class MobilenetInvertedResidual(models.mobilenet.InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()  # required for quantization

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class QuantizableMobileNetV2(nn.Module):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super().__init__()
        block = MobilenetInvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = models.mobilenet._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = models.mobilenet._make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [models.mobilenet.ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = models.mobilenet._make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(models.mobilenet.ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


@torch.jit.export
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleInvertedResidual(models.shufflenetv2.InvertedResidual):
    def __init__(self, inp, oup, stride):
        super().__init__(inp, oup, stride)
        if not hasattr(self, "branch1"):
            self.branch1 = nn.Identity();
        self.qcat = nn.quantized.FloatFunctional()

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.qcat.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = self.qcat.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class QuantizableShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=False),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleInvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleInvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=False),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x
