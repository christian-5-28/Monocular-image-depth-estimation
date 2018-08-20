import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset



class DepthDataset(Dataset):
    """
    dataset having images as inputs and targets
    """

    def images_paths(self, path):
        # traverse root directory, and list directories as dirs and files as files
        images_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
        return images_paths

    def __init__(self, input_img_path, target_img_path, input_transform=None, target_transform=None):
        """
        :param input_img_path: path for the directory containing all input images
        :param target_img_path: path for the directory containing all target images
        :param input_transform: combination of torchVision transforms applied on the input images
        :param target_transform: combination of torchVision transforms applied on the target images
        """

        self.input_img_path = input_img_path
        self.target_img_path = target_img_path

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.inputs = self.images_paths(input_img_path)
        self.targets = self.images_paths(target_img_path)

    def __getitem__(self, index):

        # using PIL image to open the specific sample
        input_img = Image.open(self.inputs[index])
        target_img = Image.open(self.targets[index])

        # applying the transforms to the sample
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)

            # check for RGBA mode in input image
            if input_img.shape[0] > 3:
                input_img = input_img[:3]

        if self.target_transform is not None:

            target_img = self.target_transform(target_img)

            # pre-processing section for depth gt stored in 16bit channel
            if np.max(target_img.numpy()) > 255:
                target_img = target_img.float() / 100
                target_img = target_img.clamp(min=0, max=40.0)

            # pre-processing section for depth gt stored in 8bit channel
            else:
                # conversion from range [0-255] to [0-39.75]
                target_img = -4.586e-09 * (target_img ** 4) + 3.382e-06 * (target_img ** 3) - 0.000105 * (
                        target_img ** 2) + 0.04239 * target_img + 0.04072

        return input_img, target_img

    def __len__(self):
        return len(self.inputs)


def to_categorical(tensor, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[tensor])


class DepthSemanticDataset(Dataset):
    """
    dataset having images as inputs and targets
    """

    def images_paths(self, path):
        # traverse root directory, and list directories as dirs and files as files
        images_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
        return images_paths

    def __init__(self, input_img_path, target_depth_path, target_semantic_path,
                 input_transform=None, depth_target_transform=None, semantic_target_transforms=None):
        """
        :param input_img_path: path for the directory containing all input images
        :param target_img_path: path for the directory containing all target images
        :param input_transform: combination of torchVision transforms applied on the input images
        :param target_transform: combination of torchVision transforms applied on the target images
        """

        self.input_img_path = input_img_path
        self.target_depth_path = target_depth_path
        self.target_semantic_path = target_semantic_path

        self.input_transform = input_transform
        self.depth_target_transform = depth_target_transform
        self.semantic_target_transforms = semantic_target_transforms

        self.inputs = self.images_paths(input_img_path)
        self.depth_targets = self.images_paths(target_depth_path)
        self.semantic_targets = self.images_paths(target_semantic_path)

    def __getitem__(self, index):

        # using PIL image to open the specific sample
        input_img = Image.open(self.inputs[index])
        target_depth_img = Image.open(self.depth_targets[index])
        target_semantic_img = Image.open(self.semantic_targets[index])

        # applying the transforms to the sample
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)

        if self.depth_target_transform is not None:
            target_depth_img = self.depth_target_transform(target_depth_img)

            # pre-processing section for depth gt stored in 16bit channel
            if np.max(target_depth_img.numpy()) > 255:
                target_depth_img = target_depth_img.float() / 100
                target_depth_img = target_depth_img.clamp(min=0, max=40.0)

            # pre-processing section for depth gt stored in 8bit channel
            else:
                # conversion from range [0-255] to [0-39.75]
                target_depth_img = -4.586e-09 * (target_depth_img ** 4) + 3.382e-06 * (
                            target_depth_img ** 3) - 0.000105 * (
                                           target_depth_img ** 2) + 0.04239 * target_depth_img + 0.04072

        if self.semantic_target_transforms is not None:
            target_semantic_img = self.semantic_target_transforms(target_semantic_img).long()

            # reshaping the semantic gt
            # target_semantic_img = target_semantic_img.view(256 * 160)
            target_semantic_img = to_categorical(target_semantic_img, 30)

        return input_img, target_depth_img, target_semantic_img

    def __len__(self):
        return len(self.inputs)
