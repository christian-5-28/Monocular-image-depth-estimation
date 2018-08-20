import pickle
import shutil

import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ..utils.utils import *
from ..utils.dataset import *
from time import strftime
import os


class DepthTrainer:
    """
    class to train and validate
    the depth model
    """

    def __init__(self, model, input_train_root_path, target_train_root_path,
                 input_test_root_path, target_test_root_path,
                 num_epochs=50, batch_size=16, learning_rate=0.001,
                 start_epoch=0, use_gpu=False, resume=None, loss_type='logRMSE'):

        self.use_gpu = use_gpu
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_epoch = start_epoch
        self.resume = resume
        self.best_val_loss = math.inf
        self.loss_type = loss_type

        if self.use_gpu:
            self.model = model.cuda()
        else:
            self.model = model

        # filtering only the weights that need to be updated with gradient
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learning_rate)

        # defining the transformation to apply on input images
        transformations = transforms.Compose([transforms.Resize((160, 256)), transforms.ToTensor()])

        # defining the transformation to apply on target images
        target_transforms = transforms.Compose([transforms.Resize((160, 256)),
                                                ImageToTensor()])

        # creating the train dataset
        self.train_dataset = DepthDataset(input_img_path=input_train_root_path,
                                          target_img_path=target_train_root_path,
                                          input_transform=transformations,
                                          target_transform=target_transforms)
        # creating the test dataset
        self.test_dataset = DepthDataset(input_img_path=input_test_root_path,
                                         target_img_path=target_test_root_path,
                                         input_transform=transformations,
                                         target_transform=target_transforms)
        print('data loaded!')

        # creating the train loader
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True
                                       )
        # creating the test loader
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False
                                      )

        # initializing th history dictionary
        self.history = {}

        # resume from checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.history = checkpoint['history']
                self.best_val_loss = checkpoint['best_val_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))

    def train_model(self, checkpoint_freq=10):

        print('training started!')
        val_losses = []
        val_accuracies = []
        train_losses = []
        train_accuracies = []

        # retrieving useful metrics from previous session
        if self.resume:
            train_losses = self.history['train_loss']
            train_accuracies = self.history['train_acc']
            val_losses = self.history['val_loss']
            val_accuracies = self.history['val_acc']

        for epoch in range(self.start_epoch, self.num_epochs):

            self.epoch = epoch

            # train for one epoch
            average_train_loss, average_train_accuracy = self.run(validate=False)

            # saving train metrics at the end of each epoch
            train_losses.append(average_train_loss)
            train_accuracies.append(average_train_accuracy)

            # saving train metrics in the history dictionary
            self.history.update({'train_loss': train_losses})
            self.history.update({'train_acc': train_accuracies})

            # evaluate model at the end of epoch on validation set
            self.model.eval()
            val_loss, val_acc = self.run(validate=True)
            self.model.train()

            # saving val metrics in the history dictionary
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            self.history.update({'val_loss': val_losses})
            self.history.update({'val_acc': val_accuracies})

            # saving the checkpoint
            if (epoch + 1) % checkpoint_freq == 0:

                # checking if we've found a better val loss
                is_best = bool(val_loss < self.best_val_loss)
                if is_best:
                    self.best_val_loss = val_loss
                    print('new best model found, val loss: %.4f' % val_loss)

                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'history': self.history,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            print('Epoch average train loss: %.4f' % average_train_loss)
            print('Epoch average train accuracy: %.4f' % average_train_accuracy)
            print('Epoch average val loss: %.4f' % val_loss)
            print('Epoch average val accuracy: %.4f' % val_acc)

        print('training concluded!')

        # Save the Trained Model and history
        model_path = "../data/model_weights/baseline_epochs" \
                     + str(self.num_epochs) + '_' + strftime("%d_%m_%Y") + \
                     "_" + strftime("%H_%M_%S") + ".pkl"

        torch.save(self.model.state_dict(), model_path)

        history_path = '../data/history/history_epochs'\
                       + str(self.num_epochs) + '_' + strftime("%d_%m_%Y") \
                       + "_" + strftime("%H_%M_%S") + ".pickle"

        with open(history_path, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, validate=False):
        """
        runs the model once on the whole dataset, back-propagation
        if validate is equal to false
        """

        if validate:
            loader = self.test_loader
        else:
            loader = self.train_loader

        losses = []
        accuracies = []

        for i, (images, targets) in enumerate(loader):

            if self.use_gpu:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                targets = Variable(targets)

            # getting predicted outputs
            outputs = self.model(images)

            # computing the loss
            loss = loss_metric(outputs, targets, loss=self.loss_type)

            losses.append(loss.data[0])

            # computing the accuracy
            accuracy = coeff_determination(outputs, targets)
            accuracies.append(accuracy)

            # back-propagation on training set
            if not validate:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, train Accuracy: %.2f'
                          % (self.epoch + 1, self.num_epochs, i + 1,
                             len(self.train_dataset) // self.batch_size,
                             loss.data[0], accuracies[-1]))

        average_accuracy = np.mean(accuracies)
        average_loss = np.mean(losses)
        return average_loss, average_accuracy

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """
        saves the current state for the checkpoint
        of the model and the best model found
        """

        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
            print('new best model saved, val loss: %.4f' % self.best_val_loss)




