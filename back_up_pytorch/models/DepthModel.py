import torch
from torch import nn


class Baseline(nn.Module):
    """
    baseline network
    """

    def __init__(self, encoder, init_weights=True):
        super(Baseline, self).__init__()

        # features attribute is the encoder part taken from the VGG19 net trained on Imagenet
        self.features = encoder

        # defining the layers of the decoder part
        self.decoder = nn.Sequential()
        self.decoder.add_module('deconv_1', module=nn.ConvTranspose2d(in_channels=512, out_channels=128,
                                                                      kernel_size=4, padding=1, stride=2))
        self.decoder.add_module('relu_1', module=nn.PReLU())

        self.decoder.add_module('deconv_2', module=nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                                      kernel_size=4, padding=1, stride=2))
        self.decoder.add_module('relu_2', module=nn.PReLU())

        self.decoder.add_module('deconv_3', module=nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                                      kernel_size=4, padding=1, stride=2))
        self.decoder.add_module('relu_3', module=nn.PReLU())

        self.decoder.add_module('deconv_4', module=nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                                      kernel_size=4, padding=1, stride=2))
        self.decoder.add_module('relu_4', module=nn.PReLU())

        self.decoder.add_module('conv_5', module=nn.Conv2d(in_channels=16, out_channels=1,
                                                           kernel_size=5, padding=2, stride=1))

        self.decoder.add_module('relu_5', module=nn.ReLU(inplace=True))

        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x


def make_encoder_layers(cfg, batch_norm=False):
    """
    reproducing the encoder part of the VGGnet,
    same procedure as in TorchVision
    """

    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]


def create_baseline(pretrained=False):
    """
    creates baseline with possibility of
    pretrained weights
    """

    weights_path = '../data/model_weights/vgg19-dcbb9e9d.pth'
    model = Baseline(make_encoder_layers(config))

    if pretrained:
        model = load_weights(model, weights_path)
    return model


def load_weights(my_model, pretrained_weights_path):
    pretrained_dict = torch.load(pretrained_weights_path)
    print('pre trained model weights loaded!')
    model_dict = my_model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    my_model.load_state_dict(model_dict)
    return my_model


def initialize_weights(net):

    # initialize the weights of the decoder part
    for m in net.decoder.modules():

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform(m.weight)

            if m.bias is not None:
                nn.init.constant(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            nn.init.constant(m.bias, 0)
