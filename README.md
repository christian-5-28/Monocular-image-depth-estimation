# EPFL Semester Project, Spring 2018. "Monocular image Depth estimation, using virtual datasets and embedding the semantic information"

## Description

Depth estimation is a fundamental tool for understanding the scene represented in an image, it underpins more complex computer vision tasks and it is of critical importance in various fields such as robotics and self-driving. The extraction of depth information from individual RGB images is of particular interest, and several works have tried to address this problem through deep learning techniques. In this study we want to investigate the possibility of training a model using only synthetic data to predict a depth map given a single image, i. e. video sequences of a virtual world reproduced by an engine. At the same time we will try to understand the impact on the performance of our network in the situation where it is optimized simultaneously on two related tasks, in our case depth estimation and semantic segmentation of a scene. We will compare two networks to verify our ideas, testing on both virtual and real data.

## Getting Started

First of all, make sure you have installed:

1. python 3.5 or higher

2. tensorFlow. Here you have the link to the official [installation guide](https://www.tensorflow.org/install/)

3. Keras. Link with all [installation steps](https://keras.io/#installation)

4. PyTorch. [installation steps](https://pytorch.org/)

5. PIL module and cv2 module

## Download the dataset

Download the Virtual kitti Dataset, in particular, download the depth and semantics groundtruths ath the following link: [Virtual Kitti Dataset](http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds)

### Brief description

1. The Keras implementation is the final and tested version of the project. In the "configuration.py" file you can change all the needed parameters for training and testing (dataset folders, hyperparameters, resuming training, etc), be careful that some of the dafault values are not valid because there are no data available (need to download the data). 

2. When the "configuration.py" file is correctly modified, you can run the "main.py" file

3. Our report of the project can be found in the "report" directory

4. the "back_up_pytorch" directory contains a pytorch implementation of the project, be aware that this version is still under testing, we share this version for the clarity of the work done.

## References

1. The keras implementation is built on the implementation of Mancini et Al's work, here at this link you can find their work: [Mancini et Al.](https://isar.unipg.it/index.php?option=com_content&view=article&id=47:j-mod2&catid=2&Itemid=188)

## Authors

* **Christian Sciuto** [christian-5-28](https://github.com/christian-5-28)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


