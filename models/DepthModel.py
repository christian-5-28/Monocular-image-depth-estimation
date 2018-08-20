from keras.models import Model
from keras.layers import Convolution2D, Input, Conv2DTranspose
from keras.applications.vgg19 import VGG19

from keras.layers.advanced_activations import PReLU

from keras.optimizers import Adam

from utils.utils import log_rmse_loss, logrmse_metric, rmse_metric


class DepthModel:
    """
    Model for the depth estimation task
    """

    def __init__(self, config):
        self.config = config

    def create_model(self):
        """
        defines the architecture of the
        model
        """

        input = Input(shape=(self.config.input_height,
                             self.config.input_width,
                             self.config.input_channel), name='input')

        # retrieve the VGG19 net already pretrained on classification task on Imagenet
        vgg19_net = VGG19(include_top=False, weights='imagenet', input_tensor=input,
                           input_shape=(self.config.input_height,
                                        self.config.input_width,
                                        self.config.input_channel
                                        ))

        # removing the fully connected layers for classification
        vgg19_net.layers.pop()
        output = vgg19_net.layers[-1].output

        # defining the decoder part of our network as a stack of
        # transposed convolution and a final convolution layer
        x = Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2))(output)
        x = PReLU()(x)
        x = Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        out = Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)

        model = Model(inputs=input, outputs=out)
        return model

    def build_model(self):
        """
        returns the model compiled with specific loss,
        metrics and optimizer
        """

        depth_net = self.create_model()

        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)
        depth_net.compile(loss=log_rmse_loss,
                          optimizer=opt,
                          metrics=[rmse_metric, logrmse_metric])

        depth_net.summary()
        return depth_net
