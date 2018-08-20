import os
from collections import OrderedDict
from glob import glob
from itertools import islice, tee

import numpy as np
import cv2

import os.path
from keras.utils import to_categorical


########################## SEQUENCE PART #####################


class DepthSequence(object):
    """
    stores all the input and depth gt
    paths in a specific directory
    """

    def __init__(self, input_directory, extension, gt_directory):

        self.input_dir = input_directory
        self.gt_dir = gt_directory

        self.image_paths = sorted(glob(os.path.join(self.input_dir, '*' + '.' + extension)))
        self.label_paths = sorted(glob(os.path.join(self.gt_dir, '*' + '.' + extension)))

    def get_image_paths(self):
        return self.image_paths

    def get_label_paths(self):
        return self.label_paths


class DepthSemanticsSequence(DepthSequence):
    """
    stores all the input paths, the depth gt
    paths and the semantic gt paths of a specific
    directory
    """

    def __init__(self, input_directory, extension, gt_directory, semantics_directory):

        self.semantics_dir = semantics_directory
        self.semantics_paths = sorted(glob(os.path.join(self.semantics_dir, '*' + '.' + extension)))
        super(DepthSemanticsSequence, self).__init__(input_directory, extension, gt_directory)

    def get_label_paths(self):
        return self.label_paths, self.semantics_paths


class Dataset(object):
    """
    dataset class that stores all the paths
    for input and gt for training set, validation
    set and test set
    """

    def __init__(self, config, name, read_semantics=False):
        """
        :param read_semantics: boolean to make distinction
        of dataset for only depth or for joint depth-semantics
        """
        self.config = config
        self.name = name
        self.read_semantics = read_semantics
        self.training_seqs = OrderedDict()
        self.test_seqs = OrderedDict()
        self.val_seqs = OrderedDict()

    def read_data(self):
        """
        for all the training directories, validation directories and
        test directories, it saves in a dictionary the paths for input
        and gt of the specific directory using the
        DepthSequence/DepthSemanticsSequence classes
        """

        if self.read_semantics or self.config.semantic_regularizer:
            for train_directory in self.config.data_train_dirs:

                train_sequence_directory = os.path.join(self.config.data_set_dir,
                                                        self.config.data_main_dir,
                                                        train_directory)

                # getting augmented data dirs --> sub directiories of the specific sequence directory
                sub_directories = os.listdir(train_sequence_directory)

                for sub_directory in sub_directories:

                    # creating the key for our dictionary
                    directory_key = os.path.join(train_directory, sub_directory)

                    # getting the complete path
                    complete_path = os.path.join(train_sequence_directory, sub_directory)

                    # saving all the paths to input and gt for the specific sub directory
                    self.training_seqs[directory_key] = DepthSemanticsSequence(os.path.join(complete_path, 'rgb'),
                                                                               self.config.img_extension,
                                                                               gt_directory=os.path.join(complete_path,
                                                                                                         'depth'),
                                                                               semantics_directory=os.path.join(
                                                                               complete_path, 'semantic'),
                                                                               )

            # repeat same procedure for the test directories
            for test_directory in self.config.data_test_dirs:
                test_sequence_directory = os.path.join(self.config.data_set_dir,
                                                       self.config.data_main_dir,
                                                       test_directory)

                # getting augmented data dirs --> sub directiories of the specific sequence directory
                sub_directories = os.listdir(test_sequence_directory)

                for sub_directory in sub_directories:

                    # creating the key for our dictionary
                    directory_key = os.path.join(test_directory, sub_directory)

                    # getting the complete path
                    complete_path = os.path.join(test_sequence_directory, sub_directory)

                    self.test_seqs[directory_key] = DepthSemanticsSequence(os.path.join(complete_path, 'rgb'),
                                                                           self.config.img_extension,
                                                                           gt_directory=os.path.join(complete_path,
                                                                                                     'depth'),
                                                                           semantics_directory=os.path.join(complete_path,
                                                                                                            'semantic'),
                                                                       )
            # same procedure for the validation directories
            for validation_directory in self.config.data_val_dirs:
                validation_sequence_directory = os.path.join(self.config.data_set_dir,
                                                             self.config.data_main_dir,
                                                             validation_directory)

                # getting augmented data dirs
                sub_directories = os.listdir(validation_sequence_directory)

                for sub_directory in sub_directories:
                    directory_key = os.path.join(validation_directory, sub_directory)
                    complete_path = os.path.join(validation_sequence_directory, sub_directory)

                    self.val_seqs[directory_key] = DepthSemanticsSequence(os.path.join(complete_path, 'rgb'),
                                                                          self.config.img_extension,
                                                                          gt_directory=os.path.join(complete_path,
                                                                                                    'depth'),
                                                                          semantics_directory=os.path.join(complete_path,
                                                                                                           'semantic'),
                                                                          )
            return

        # case of when we deal only with the depth estimation task
        else:
            for train_directory in self.config.data_train_dirs:

                train_sequence_directory = os.path.join(self.config.data_set_dir,
                                                        self.config.data_main_dir,
                                                        train_directory)

                # getting sub directories
                sub_directories = os.listdir(train_sequence_directory)

                for sub_directory in sub_directories:

                    directory_key = os.path.join(train_directory, sub_directory)
                    complete_path = os.path.join(train_sequence_directory, sub_directory)

                    self.training_seqs[directory_key] = DepthSequence(os.path.join(complete_path, 'rgb'),
                                                                      self.config.img_extension,
                                                                      gt_directory=os.path.join(complete_path, 'depth'),
                                                                      )

            for test_directory in self.config.data_test_dirs:
                test_sequence_directory = os.path.join(self.config.data_set_dir,
                                                       self.config.data_main_dir,
                                                       test_directory)

                # getting augmented data dirs
                sub_directories = os.listdir(test_sequence_directory)

                for sub_directory in sub_directories:

                    directory_key = os.path.join(test_directory, sub_directory)
                    complete_path = os.path.join(test_sequence_directory, sub_directory)

                    self.test_seqs[directory_key] = DepthSequence(os.path.join(complete_path, 'rgb'),
                                                                  self.config.img_extension,
                                                                  gt_directory=os.path.join(complete_path, 'depth'),
                                                                  )

            for validation_directory in self.config.data_val_dirs:
                validation_sequence_directory = os.path.join(self.config.data_set_dir,
                                                             self.config.data_main_dir,
                                                             validation_directory)

                # getting augmented data dirs
                sub_directories = os.listdir(validation_sequence_directory)

                for sub_directory in sub_directories:

                    directory_key = os.path.join(validation_directory, sub_directory)
                    complete_path = os.path.join(validation_sequence_directory, sub_directory)

                    self.val_seqs[directory_key] = DepthSequence(os.path.join(complete_path, 'rgb'),
                                                                 self.config.img_extension,
                                                                 gt_directory=os.path.join(complete_path, 'depth'),
                                                                 )
            return


class DataLoader:
    """
    class to load and pre-process batches of samples
    for training, validation and test sequences
    """

    def __init__(self, dataset, shuffle_data=True):
        self.dataset = dataset
        self.dataset.read_data()
        self.shuffle = shuffle_data
        self.config = dataset.config

        # create training and test set paths
        self.training_set = self.generate_data(self.dataset.training_seqs)
        self.test_set = self.generate_data(self.dataset.test_seqs)

        # creating val set paths
        self.validation_set = self.generate_data(self.dataset.val_seqs)

    def generate_data(self, sequences):
        """
        for each sequence stored in the dictionary it creates
        samples in the form of a tuple (input, gt) and finally
        returns the all set of samples

        :param sequences: dictionary of input/gt paths
        :return:
        """

        self.width = self.dataset.config.input_width
        self.height = self.dataset.config.input_height

        img_sets = []
        for seq in sequences:

            # getting the value from the dictionary
            curr_seq = sequences[seq]

            # creating the samples from the specific sequence
            seq_set = self.get_image_set(curr_seq)

            # collecting the samples of all the sequences in one list
            img_sets.extend(seq_set)

        return img_sets

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)

    def get_image_set(self, sequence):
        """
        creates all the samples as tuples (input, gt)
        for the specific sequence
        """

        samples = []

        # distinguishes between only depth and jointly depth/semantics
        if self.dataset.read_semantics or self.config.semantic_regularizer:
            for input_img_path, depth_gt_path, semantic_gt_path in zip(self.nwise(sequence.get_image_paths(), 1),
                                                                       self.nwise(sequence.get_label_paths()[0], 1),
                                                                       self.nwise(sequence.get_label_paths()[1], 1)):

                # adding the new sample to the list
                if self.config.semantic_regularizer:
                    samples.append(([input_img_path[0], semantic_gt_path[0]], [depth_gt_path[0]]))
                else:
                    samples.append((input_img_path[0], [depth_gt_path[0], semantic_gt_path[0]]))

            return samples

        else:
            for input_img_path, depth_gt_path in zip(self.nwise(sequence.get_image_paths(), 1),
                                                     self.nwise(sequence.get_label_paths(), 1)):

                # adding the new sample to the list
                samples.append((input_img_path[0], [depth_gt_path[0]]))
            return samples

    def input_transform(self, path):
        """
        reads and apply a resize on the input image
        of the specific path
        """
        if self.config.semantic_regularizer:
            input_path = path[0]
            semantic_path = path[1]

            rgb_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            rgb_img = cv2.resize(rgb_img, (256, 160), cv2.INTER_LINEAR)

            semantic_img = cv2.imread(semantic_path, cv2.IMREAD_ANYDEPTH)
            semantic_img = cv2.resize(semantic_img, (256, 160), cv2.INTER_NEAREST)

            return [rgb_img, semantic_img]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 160), cv2.INTER_LINEAR)
        return img

    def label_transform(self, labels_paths):
        """
        reads and applies transformations on
        the groundtruth values of the specific paths
        """

        depth_gt_path = labels_paths[0]

        # read the depth gt at the specific path
        depth_label = cv2.imread(depth_gt_path, cv2.IMREAD_ANYDEPTH)

        # resize the depth gt
        depth_label = cv2.resize(depth_label, (256, 160), cv2.INTER_NEAREST)

        # creating a dictionary to manage easily the joint labels
        labels = {}
        labels["depth"] = np.expand_dims(depth_label, 2)

        # storing the semantic gt if the dataset is for the joint depth/semantics
        if self.dataset.read_semantics:

            semantic_gt_path = labels_paths[1]

            semantic_label = cv2.imread(semantic_gt_path, cv2.IMREAD_ANYDEPTH)
            semantic_label = cv2.resize(semantic_label, (256, 160), cv2.INTER_NEAREST)

            # reshaping the semantic gt
            semantic_label = np.reshape(semantic_label, newshape=(256 * 160))
            
            # using "to_categorical" in order to have the gt
            # compliant  for the keras cross-entropy loss
            labels["semantic"] = to_categorical(semantic_label, 30)

        return labels

    def data_generator(self, samples_set):
        """
        samples generator compliant
        for keras implementation
        """

        if self.shuffle:
            np.random.seed(self.config.random_seed)
            np.random.shuffle(samples_set)

        curr_batch = 0

        while 1:

            if (curr_batch + 1) * self.dataset.config.batch_size > len(samples_set):
                np.random.shuffle(samples_set)
                curr_batch = 0

            inputs_x = []
            gt_y = []

            # creating a batch of samples
            for sample in samples_set[curr_batch *
                                      self.dataset.config.batch_size: (curr_batch + 1) *
                                                                            self.dataset.config.batch_size]:

                features_path = sample[0]
                labels_path = sample[1]

                # loading and transform the input image
                input_img = self.input_transform(features_path)

                # loading and transform the labels
                labels = self.label_transform(labels_paths=labels_path)

                inputs_x.append(input_img)
                gt_y.append(labels)

            # pre-processing the data for the model
            inputs_x, gt_y = self.prepare_data_for_model(inputs_x, gt_y)

            curr_batch += 1

            yield inputs_x, gt_y

    def test_data_generator(self):
        """
        data generator for test sequences
        for single sample
        """

        current_sample = 0
        len_test_set = len(self.test_set)

        while current_sample < len_test_set:

            inputs_x = []
            gt_y = []

            features_path = self.test_set[current_sample][0]
            labels_path = self.test_set[current_sample][1]

            # loading and transform the input image
            input_img = self.input_transform(features_path)

            # loading and transform the labels
            labels = self.label_transform(labels_paths=labels_path)

            inputs_x.append(input_img)
            gt_y.append(labels)

            # preparing the values for the model
            x_test, y_test = self.prepare_data_for_model(inputs_x, gt_y)
            # x_test = np.expand_dims(x_test, axis=0)
            current_sample += 1

            yield x_test, y_test

    def prepare_data_for_model(self, input_x, gt_y):
        """
        applies pre-processing on inputs and gt labels
        returning values as numpy arrays
        """
        if self.config.semantic_regularizer:
            rgb_inputs = [batch_sample[0] for batch_sample in input_x]
            semantic_inputs = [batch_sample[1] for batch_sample in input_x]
            semantic_inputs = np.asarray(semantic_inputs).astype(np.uint8)
            features = np.asarray(rgb_inputs).astype('float32') / 255
            inputs = [features, semantic_inputs]
        else:
            features = np.asarray(input_x).astype('float32') / 255
            inputs = features

        labels_depth = np.zeros(shape=(features.shape[0],features.shape[1],features.shape[2],1), dtype=np.float32)
        labels_semantics = np.zeros(shape=(features.shape[0], features.shape[1] * features.shape[2], 30,), dtype=np.uint8)
        i = 0
        for elem in gt_y:

            label = []

            elem["depth"] = np.asarray(elem["depth"]).astype(np.float32)

            # pre-processing section for depth gt stored in 16bit channel
            if np.max(elem["depth"]) > 255:
                elem["depth"] = elem["depth"] / 100
                elem["depth"] = elem["depth"].clip(min=0, max=40.0)

            # pre-processing section for depth gt stored in 8bit channel
            else:

                # conversion from range [0-255] to [0-39.75]
                elem["depth"] = -4.586e-09 * (elem["depth"] ** 4) + 3.382e-06 * (elem["depth"] ** 3) - 0.000105 * (
                            elem["depth"] ** 2) + 0.04239 * elem["depth"] + 0.04072

            labels_depth[i, :, :, :] = elem["depth"]

            # section for the semantics gt
            if self.dataset.read_semantics and not self.config.semantic_regularizer:
                labels_semantics[i, :, :] = np.asarray(elem["semantic"]).astype(np.uint8)
                label.append(labels_depth)
                label.append(labels_semantics)
            else:
                label = labels_depth
            i += 1

        return inputs, label
