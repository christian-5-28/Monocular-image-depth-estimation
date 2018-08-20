import pickle
import shutil

import math

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ..utils.utils import *
from ..utils.dataset import *
from timeit import default_timer as timer
from time import strftime
import os


class JointTrainer:
    """
    class to train and validate the model
    with parallel depth estimation and semantic
    segmentation
    """

    def __init__(self, model,
                 input_train_root_path,
                 depth_train_root_path,
                 input_test_root_path,
                 depth_test_root_path,
                 semantics_train_root_path,
                 semantics_test_root_path,
                 num_epochs=50,
                 batch_size=16,
                 learning_rate=0.0001,
                 start_epoch=0,
                 use_gpu=False,
                 resume=None,
                 loss_type='logRMSE'):

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

        transformations = transforms.Compose([transforms.Resize((64, 256)), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        # transforms for the depth gt
        depth_target_transforms = transforms.Compose([transforms.Resize((64, 256)),
                                                     transforms.ToTensor()])

        # transforms for semantic gt
        semantic_target_transforms = transforms.Compose([transforms.Resize((64, 256)),
                                                         ImageToTensor()])

        self.train_dataset = DepthSemanticDataset(input_img_path=input_train_root_path,
                                                  target_depth_path=depth_train_root_path,
                                                  target_semantic_path=semantics_train_root_path,
                                                  input_transform=transformations,
                                                  depth_target_transform=depth_target_transforms,
                                                  semantic_target_transforms=semantic_target_transforms
                                                  )

        self.test_dataset = DepthSemanticDataset(input_img_path=input_test_root_path,
                                                 target_depth_path=depth_test_root_path,
                                                 target_semantic_path=semantics_test_root_path,
                                                 input_transform=transformations,
                                                 depth_target_transform=depth_target_transforms,
                                                 semantic_target_transforms=semantic_target_transforms
                                                 )

        print('data loaded!')

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True
                                       )

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      shuffle=True
                                      )
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
        depth_val_losses = []
        depth_val_accuracies = []
        depth_train_losses = []
        depth_train_accuracies = []

        semantic_val_losses = []
        semantic_val_accuracies = []
        semantic_train_losses = []
        semantic_train_accuracies = []

        if self.resume:
            depth_train_losses = self.history['depth_train_loss']
            depth_train_accuracies = self.history['depth_train_acc']
            depth_val_losses = self.history['depth_val_loss']
            depth_val_accuracies = self.history['depth_val_acc']

            semantic_train_losses = self.history['semantic_train_loss']
            semantic_train_accuracies = self.history['semantic_train_acc']
            semantic_val_losses = self.history['semantic_val_loss']
            semantic_val_accuracies = self.history['semantic_val_acc']

        verbose = 1

        for epoch in range(self.start_epoch, self.num_epochs):

            epoch_depth_train_losses = []
            epoch_depth_train_accuracies = []
            epoch_semantic_train_losses = []

            start_time = timer()
            for i, (images, depth_targets, semantic_targets) in enumerate(self.train_loader):

                if i / self.batch_size % verbose == 0:
                    start_time = timer()

                if self.use_gpu:
                    images = Variable(images.cuda())
                    depth_targets = Variable(depth_targets.cuda())
                    semantic_targets = Variable(semantic_targets.cuda())
                else:
                    images = Variable(images)
                    depth_targets = Variable(depth_targets)
                    semantic_targets = Variable(semantic_targets.long())

                # Forward + Backward + Optimize
                # getting predicted outputs
                depth_outputs, semantic_outputs = self.model(images)

                # computing the depth training loss
                depth_train_loss = loss_metric(depth_outputs, depth_targets, loss=self.loss_type)
                epoch_depth_train_losses.append(depth_train_loss.data[0])

                # computing the semantic training loss
                semantic_loss = loss_metric(semantic_outputs,
                                            semantic_targets.squeeze(),
                                            loss='cross_entropy')

                epoch_semantic_train_losses.append(semantic_loss.data[0])

                # computing the depth training accuracy
                train_depth_accuracy = coeff_determination(depth_outputs, depth_targets)
                epoch_depth_train_accuracies.append(train_depth_accuracy)

                final_loss = depth_train_loss + semantic_loss

                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

                if i / self.batch_size % verbose == 0:

                    print('Iteration number: ', i / self.batch_size * (epoch + 1))
                    print('Step to the end: ', (len(self.train_dataset.inputs) - i * (epoch + 1)) / self.batch_size)

                    print('Epoch [%d/%d], Iter [%d/%d] depth Loss: %.4f, semantic loss: %.4f, depth Accuracy: %.2f'
                          % (epoch + 1, self.num_epochs, i + 1,
                             len(self.train_dataset) // self.batch_size,
                             depth_train_loss.data[0], semantic_loss.data[0],
                             epoch_depth_train_accuracies[-1]))


            # saving metrics at the end of each epoch
            depth_avg_train_loss = np.mean(epoch_depth_train_losses)
            depth_train_losses.append(depth_avg_train_loss)
            depth_avg_train_acc = np.mean(epoch_depth_train_accuracies)
            depth_train_accuracies.append(depth_avg_train_acc)

            semantic_avg_train_loss = np.mean(epoch_semantic_train_losses)
            semantic_train_losses.append(semantic_avg_train_loss)
            # computing the semantic training accuracy
            train_semantic_accuracy = self.check_semantic_accuracy(use_gpu=self.use_gpu)
            semantic_train_accuracies.append(train_semantic_accuracy)

            # saving learning data in the history dictionary
            self.history.update({'depth_train_loss': depth_train_losses})
            self.history.update({'depth_train_acc': depth_train_accuracies})
            self.history.update({'semantic_train_loss': semantic_train_losses})
            self.history.update({'semantic_train_acc': semantic_train_accuracies})

            self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

            average_depth_loss, average_depth_accuracy, average_semantic_loss, val_semantic_accuracy = self.validate()

            self.model.train()

            depth_val_losses.append(average_depth_loss)
            depth_val_accuracies.append(average_depth_accuracy)
            self.history.update({'depth_val_loss': depth_val_losses})
            self.history.update({'depth_val_acc': depth_val_accuracies})

            semantic_val_losses.append(average_semantic_loss)
            semantic_val_accuracies.append(val_semantic_accuracy)
            self.history.update({'semantic_val_loss': semantic_val_losses})
            self.history.update({'semantic_val_acc': semantic_val_accuracies})

            # saving the checkpoint
            if (epoch + 1) % checkpoint_freq == 0:

                is_best = bool(average_depth_loss < self.best_val_loss)
                if is_best:
                    self.best_val_loss = average_depth_loss

                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'history': self.history,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            print('Epoch average depth train loss: %.4f' % depth_avg_train_loss)
            print('Epoch average depth train accuracy: %.4f' % depth_avg_train_acc)
            print('Epoch average depth val loss: %.4f' % average_depth_loss)
            print('Epoch average depth val accuracy: %.4f' % average_depth_accuracy)

            print('Epoch average semantic train loss: %.4f' % semantic_avg_train_loss)
            print('Epoch average semantic train accuracy: %.4f' % train_semantic_accuracy)
            print('Epoch average semantic val loss: %.4f' % average_semantic_loss)
            print('Epoch average semantic val accuracy: %.4f' % val_semantic_accuracy)

            end_time = timer()
            time_elapsed = end_time - start_time
            predicted_time = time_elapsed * self.num_epochs
            time_remaining = predicted_time - ((epoch + 1) * time_elapsed)
            m, s = divmod(time_remaining, 60)
            h, m = divmod(m, 60)
            print("time remaining: %d:%02d:%02d" % (h, m, s))

        print('training concluded!')

        # Save the Trained Model and history
        model_path = "../data/model_weights/jointModel_epochs" \
                     + str(self.num_epochs) + '_' + strftime("%d_%m_%Y") + \
                     "_" + strftime("%H_%M_%S") + ".pkl"

        torch.save(self.model.state_dict(), model_path)

        history_path = '../data/history/joint_history_epochs' \
                       + str(self.num_epochs) + '_' + strftime("%d_%m_%Y") +\
                       "_" + strftime("%H_%M_%S") + ".pickle"

        with open(history_path, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def validate(self):
        depth_validation_losses = []
        depth_validation_accuracies = []
        semantic_validation_losses = []

        for images, depth_targets, semantic_targets in self.test_loader:

            if self.use_gpu:
                images = Variable(images.cuda())
                depth_targets = Variable(depth_targets.cuda())
                semantic_targets = Variable(semantic_targets.cuda())
            else:
                images = Variable(images)
                depth_targets = Variable(depth_targets)
                semantic_targets = Variable(semantic_targets)

            # getting predicted outputs
            depth_outputs, semantic_outputs = self.model(images)

            # depth validation loss and accuracy
            depth_validation_losses.append(loss_metric(depth_outputs, depth_targets, loss=self.loss_type).data[0])
            depth_validation_accuracies.append(coeff_determination(depth_outputs, depth_targets))

            # computing the semantic validation loss
            semantic_loss = loss_metric(semantic_outputs,
                                        semantic_targets.squeeze(dim=1),
                                        loss='cross_entropy')

            semantic_validation_losses.append(semantic_loss.data[0])

        # computing the semantic validation accuracy
        val_semantic_accuracy = self.check_semantic_accuracy(self.use_gpu, training=False)

        average_depth_accuracy = np.mean(depth_validation_accuracies)
        average_depth_loss = np.mean(depth_validation_losses)
        average_semantic_loss = np.mean(semantic_validation_losses)

        return average_depth_loss, average_depth_accuracy, average_semantic_loss, val_semantic_accuracy

    def save_checkpoint(self, state, is_best, filename='jointModel_checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'jointModel_model_best.pth.tar')

    def check_semantic_accuracy(self, use_gpu, training=True):
        """
        Check the accuracy of the model.
        """

        num_correct, num_samples = 0, 0

        if training:
            loader = self.train_loader
        else:
            loader = self.test_loader

        for images, depth_targets, semantic_targets in loader:

            images = Variable(images.cuda(), volatile=True) if use_gpu else Variable(images)
            semantic_targets = semantic_targets.squeeze()

            pred_depth, pred_labels = self.model(images)

            # Returns the maximum value of each row of the input tensor in the given dimension dim.
            # The second return value is the index location of each maximum value found (argmax).

            batch_size = pred_labels.size()[0]

            for image_index in range(batch_size):
                pred_label = pred_labels[image_index]
                _, preds = pred_label.data.cpu().max(0)

                # Computing pixel-wise accuracy
                sem_target = semantic_targets[image_index]
                temp_correct = (preds.long() == sem_target.long()).sum()
                num_correct += temp_correct
                num_pixels = preds.numel()
                num_samples += preds.numel()

                if temp_correct >= num_pixels:
                    print('acc_error')

        if num_correct > num_samples:
            print('acc_error')

        acc = float(num_correct) / num_samples

        return acc

