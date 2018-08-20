from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Reshape, Convolution2D, Conv2DTranspose, PReLU, Softmax, Permute
from keras.optimizers import Adam

from models.DepthModel import DepthModel
from utils.utils import logrmse_metric, log_rmse_loss


class DsNet(DepthModel):
    """
    model that solves jointly two tasks:
    depth estimation and semantic segmentation
    """

    def build_model(self):
        """
        adds the semantic branch end returns the
        compiled model with losses, metrics and optimizer
        """

        # creating the encoder and the depth branch
        depth_model = self.create_model()

        # creating the semantic branch having as input the
        # output of the encoder part
        output = depth_model.layers[-10].output

        x = Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2))(output)
        x = PReLU()(x)
        x = Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Convolution2D(30, (5, 5), padding="same", activation="relu")(x)

        # reshaping the signal to prepare it for the Softmax activation function
        x = Reshape(target_shape=(30, self.config.input_width * self.config.input_height))(x)
        x = Permute((2, 1))(x)
        semantic_output = Softmax(name="semantic_output")(x)

        model = Model(inputs=depth_model.inputs[0], outputs=[depth_model.outputs[0], semantic_output])

        # compiling the model
        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)
        model.compile(loss={'depth_output': log_rmse_loss,
                            'semantic_output': categorical_crossentropy},
                      optimizer=opt,
                      metrics={'depth_output': [logrmse_metric], 'semantic_output': [categorical_accuracy]},
                      loss_weights=[1.0, 1.0])

        model.summary()
        return model




