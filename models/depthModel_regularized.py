from keras.models import Model
from keras.layers import Convolution2D, Input, Conv2DTranspose
from keras.applications.vgg19 import VGG19
import keras.backend as K

from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam

from utils.utils import log_rmse_loss, logrmse_metric, rmse_metric


class DepthModelRegularized:
    """
    Model for the depth estimation task
    """

    def __init__(self, config):
        self.config = config

    def custom_loss_wrapper(self, input_tensor, mask_coeff=10, classes=(19, 20, 21, 26, 27)):
        """
        wrapper fucntion in order to have a valid
        behavior with the Keras API for the loss function
        :param input_tensor: the semantic groundtruth
        :param mask_coeff: value used to weight differently the error on specific semantic classes
        :param classes: semantic class ids to consider for the mask (the id values follow
                        the same values of the CityScapes dataset)
        """

        def custom_loss(y_true, y_pred):
            shape = K.int_shape(input_tensor)
            shape = (shape[1], shape[2])
            mask = K.zeros(shape=shape)

            # the mask is created considering all the class ids proposed
            for class_id in classes:
                mask += K.cast(K.equal(input_tensor, class_id), K.floatx())

            # computing the logrmse loss
            loss = log_rmse_loss(y_true=y_true, y_pred=y_pred)

            # return the loss with a larger error on the pixels selected using the mask
            return loss + mask_coeff * (loss * mask)

        return custom_loss

    def create_model(self):
        """
        defines the architecture of the
        model
        """

        rgb_input = Input(shape=(self.config.input_height,
                                 self.config.input_width,
                                 self.config.input_channel),
                          name='rgb_input')

        semantic_input = Input(shape=(self.config.input_height,
                                      self.config.input_width),
                               name='semantic_input')

        # retrieve the VGG19 net already pretrained on classification task on Imagenet
        vgg19_net = VGG19(include_top=False, weights='imagenet', input_tensor=rgb_input,
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

        # model with two inputs: rgb and the semantic gt
        model = Model(inputs=[rgb_input, semantic_input], outputs=out)
        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)

        # using the custom loss wrapper
        model.compile(loss=self.custom_loss_wrapper(input_tensor=semantic_input,
                                                    mask_coeff=self.config.mask_coeff),
                      optimizer=opt,
                      metrics=[rmse_metric, logrmse_metric])
        return model

    def build_model(self):
        """
        returns the model compiled with specific loss,
        metrics and optimizer
        """
        depth_net = self.create_model()
        depth_net.summary()
        return depth_net
