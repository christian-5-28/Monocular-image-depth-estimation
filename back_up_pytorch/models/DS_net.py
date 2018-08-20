from .DepthModel import make_encoder_layers, load_weights
from torch import nn


class JointNet(nn.Module):

    def __init__(self, encoder_net, init_weights=True):
        super(JointNet, self).__init__()

        # common encoding part
        self.features = encoder_net

        # depth-estimation branch
        self.depth_branch = nn.Sequential()
        self.depth_branch.add_module('deconv_1', module=nn.ConvTranspose2d(in_channels=512, out_channels=128,
                                                                           kernel_size=4, padding=1, stride=2))
        self.depth_branch.add_module('relu_1', module=nn.PReLU())

        self.depth_branch.add_module('deconv_2', module=nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                                           kernel_size=4, padding=1, stride=2))
        self.depth_branch.add_module('relu_2', module=nn.PReLU())

        self.depth_branch.add_module('deconv_3', module=nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                                           kernel_size=4, padding=1, stride=2))
        self.depth_branch.add_module('relu_3', module=nn.PReLU())

        self.depth_branch.add_module('deconv_4', module=nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                                           kernel_size=4, padding=1, stride=2))
        self.depth_branch.add_module('relu_4', module=nn.PReLU())

        self.depth_branch.add_module('conv_5', module=nn.Conv2d(in_channels=16, out_channels=1,
                                                                kernel_size=5, padding=2, stride=1))

        self.depth_branch.add_module('relu_5', module=nn.ReLU(inplace=True))


        # semantic_seg part
        self.semantic_branch = nn.Sequential()

        self.semantic_branch.add_module('deconv_1', module=nn.ConvTranspose2d(in_channels=512, out_channels=128,
                                                                              kernel_size=4, padding=1, stride=2))
        self.semantic_branch.add_module('relu_1', module=nn.PReLU())

        self.semantic_branch.add_module('deconv_2', module=nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                                              kernel_size=4, padding=1, stride=2))
        self.semantic_branch.add_module('relu_2', module=nn.PReLU())

        self.semantic_branch.add_module('deconv_3', module=nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                                              kernel_size=4, padding=1, stride=2))
        self.semantic_branch.add_module('relu_3', module=nn.PReLU())

        self.semantic_branch.add_module('deconv_4', module=nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                                              kernel_size=4, padding=1, stride=2))
        self.semantic_branch.add_module('relu_4', module=nn.PReLU())

        self.depth_branch.add_module('conv_5', module=nn.Conv2d(in_channels=16, out_channels=30,
                                                                kernel_size=5, padding=2, stride=1))

        self.depth_branch.add_module('relu_5', module=nn.ReLU(inplace=True))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features_embedded = self.features(x)

        # forwarding on the depth branch
        predicted_depth = self.depth_branch(features_embedded)

        # forwarding on the semantic segmentation branch
        predicted_labels = self.semantic_branch(features_embedded)

        return predicted_depth, predicted_labels

    def _initialize_weights(self):

        # weights initialization of the two branches
        self.init_weights_sequential(self.depth_branch)
        self.init_weights_sequential(self.semantic_branch)

    def init_weights_sequential(self, seq):

        for m in seq.modules():

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


config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def create_join_net(pretrained=False):

    model = JointNet(make_encoder_layers(config))
    if pretrained:
        model = load_weights(model, '../data/model_weights/vgg19-dcbb9e9d.pth')
    return model
