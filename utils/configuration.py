# -*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--data_set_dir', type=str, default='/data/vkitti_full_dataset')

data_arg.add_argument('--data_main_dir', type=str, default='')
data_arg.add_argument('--data_train_dirs', type=eval, nargs='+', default=['0001', '0018', '0020'])
data_arg.add_argument('--data_test_dirs', type=eval, nargs='+', default=['0002'])
data_arg.add_argument('--data_val_dirs', type=eval, nargs='+', default=['0006'])

data_arg.add_argument('--input_height', type=int, default=160)
data_arg.add_argument('--input_width', type=int, default=256)

data_arg.add_argument('--input_channel', type=int, default=3)
data_arg.add_argument('--img_extension', type=str, default="png")

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--exp_name', type=str, default='NAME_OF_EXPERIMENT')
train_arg.add_argument('--preload_ram', type=str2bool, default=False)
train_arg.add_argument('--max_step', type=int, default=10000, help='')
train_arg.add_argument('--batch_size', type=int, default=32, help='')
train_arg.add_argument('--buffer_size', type=int, default=25600, help='')
train_arg.add_argument('--num_epochs', type=int, default=60, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=1e-5, help='')
train_arg.add_argument('--read_semantics', type=str2bool, default=False, help='')
train_arg.add_argument('--semantic_regularizer', type=str2bool, default=False, help='')
train_arg.add_argument('--mask_coeff', type=float, default=10, help='')

train_arg.add_argument('--weights_path', type=str,
                       default="/directory/to/the/model_weights.hdf5")

train_arg.add_argument('--resume_training', type=str2bool, default=False)
train_arg.add_argument('--initial_epoch', type=int, default=1)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--predict_qualitative_path', type=str, default='data/qualitative/predict')
misc_arg.add_argument('--gt_qualitative_path', type=str, default='data/qualitative/gt')
misc_arg.add_argument('--log_step', type=int, default=10, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')  # the model checkpoints are saved in this directory
misc_arg.add_argument('--debug', type=str2bool, default=True)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=0.5)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
