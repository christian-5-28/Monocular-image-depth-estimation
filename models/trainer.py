import os
import math
import time
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2

from utils.utils import BatchCallback, logrmse_metric_np, rmse_metric_np


class Trainer:
    """
    Handles the training, validation and testing session for a specific
    model on a specific dataset (through a data loader)
    """

    def __init__(self, model, data_loader, config, rng):
        self.model = model.build_model()
        self.data_loader = data_loader
        self.config = config

        self.rng = rng

        # directory where the logs will be saved
        self.model_dir = config.log_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction

        # value to specify the interval of epochs
        # when the logs have to be saved
        self.log_step = config.log_step
        self.max_step = config.max_step

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

    def train(self, initial_epoch=0):

        number_batches_per_epoch = int(math.floor(len(self.data_loader.training_set) / self.config.batch_size))
        validation_steps = int(math.floor(len(self.data_loader.validation_set) / self.config.batch_size))

        print("Samples per epoch: {}".format(number_batches_per_epoch))

        # callbacks section
        # defining the tensorboard callback specifying the tensorboard directory
        keras_tensorboard = TensorBoard(log_dir=self.config.tensorboard_dir,
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=False)

        # creating the model checkpoint that saves model weights every log step
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),
            monitor='loss', verbose=2,
            save_best_only=False, save_weights_only=False, mode='auto', period=self.config.log_step)

        # callback for printing values at the end of each batch
        batch_callback = BatchCallback()

        # lr_scheduler
        '''
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                                      patience=5, min_lr=10e-7, verbose=1)
        '''

        # taking the time of the training session
        t0 = time.time()

        # starting the training session
        self.model.fit_generator(generator=self.data_loader.data_generator(self.data_loader.training_set),
                                 steps_per_epoch=number_batches_per_epoch,
                                 callbacks=[batch_callback, model_checkpoint, keras_tensorboard], # reduce_lr]
                                 validation_data=self.data_loader.data_generator(self.data_loader.validation_set),
                                 validation_steps=validation_steps,
                                 epochs=self.config.num_epochs,
                                 verbose=2,
                                 initial_epoch=initial_epoch)
        t1 = time.time()

        print("Training completed in " + str(t1 - t0) + " seconds")

    def resume_training(self):
        """
        restarts training for a pretrained model
        from a specific epoch
        :return:
        """
        print("resuming training from weights file: ", self.config.weights_path)
        self.model.load_weights(self.config.weights_path)
        self.train(self.config.initial_epoch)

    def test(self, show_figure=False, save_figures=False, is_real_kitti=False):
        """
        :param show_figure: boolean for qualitative results
        :param is_real_kitti: boolean to handle the boolean mask on the real Kitti dataset
        """

        # loading the model weights
        self.model.load_weights(self.config.weights_path)
        rmse_metric_accumulator = 0
        logrmse_metric_accumulator = 0

        for index, (x_test, y_test) in enumerate(self.data_loader.test_data_generator()):

            print('Testing img: ', index)

            # getting the predicted output
            output = self.model.predict(x_test)

            # getting only the depth output
            if self.config.read_semantics:
                depth_output = output[0]
                depth_gt = y_test[0]

            else:
                depth_output = output
                depth_gt = y_test

            # creating the mask on the real kitti samples
            # in order to evaluate only valid pixels (pixel != 0)
            if is_real_kitti:
                y_true = depth_gt.reshape(1, -1)
                y_pred = depth_output.reshape(1, -1)

                mask = y_true != 0
                y_true = y_true[mask]
                y_pred = y_pred[mask]
            else:
                y_true = depth_gt
                y_pred = depth_output

            # saving the metrics
            temp_rmse = rmse_metric_np(y_true, y_pred)
            rmse_metric_accumulator += temp_rmse

            temp_logrmse = logrmse_metric_np(y_true, y_pred)
            logrmse_metric_accumulator += temp_logrmse
            print('Rmse: ', temp_rmse)
            print('Log RMSE: ', temp_logrmse)

            # dealing with the qualitative results
            if show_figure or save_figures:
                self.handle_qualitative(depth_output=depth_output,
                                        depth_gt=depth_gt,
                                        x_test=x_test,
                                        image_number=index,
                                        rmse=temp_rmse,
                                        logrmse=temp_logrmse,
                                        show_figure=show_figure,
                                        save_figure=save_figures)

        print('Average rmse: ')
        print(rmse_metric_accumulator / len(self.data_loader.test_set))

        print('Average log rmse: ')
        print(logrmse_metric_accumulator / len(self.data_loader.test_set))

    def handle_qualitative(self, depth_output, depth_gt, x_test,
                           image_number, rmse, logrmse, show_figure, save_figure):
        if len(depth_output.shape) == 4:
            depth = depth_output[0, :, :, :]
        if len(depth_gt.shape) == 4:
            gt = depth_gt[0, ...]
        else:
            gt = depth_gt.copy()

        # creating the predicted colormap making sure to avoid overflows
        output_cv = (((depth - np.min(depth)) / np.max(depth)) * 255)
        output_cv = output_cv.astype("uint8")
        depth_jet = cv2.applyColorMap(output_cv, cv2.COLORMAP_JET)

        # creating the gt colormap
        real_cv = (((gt - np.min(gt)) / np.max(gt)) * 255)
        real_cv = real_cv.astype("uint8")
        real_jet = cv2.applyColorMap(real_cv, cv2.COLORMAP_JET)

        if show_figure:
            cv2.imshow("rgb", x_test[0, :, :, :])
            cv2.imshow("Predicted Depth", depth_jet)
            cv2.imshow("Real Depth", real_jet)
            cv2.waitKey(10)

        if save_figure:
            # path for predicted images
            predict_path = self.config.predict_qualitative_path
            name = str(image_number) + '_rmse_' + str(rmse) + '_logrmse_' + str(logrmse) + '_full.png'
            cv2.imwrite(os.path.join(predict_path, name), depth_jet)

            # path for gt map
            gt_path = self.config.gt_qualitative_path
            real_name = str(image_number) + '_real_full.png'
            cv2.imwrite(os.path.join(gt_path, real_name), real_jet)
