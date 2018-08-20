from .models.DS_net_trainer import *
from .models.DS_net import *
from .models.depth_trainer import *
from .models.DepthModel import *
import torch
import sys


input_train_root_path, depth_train_root_path, input_test_root_path,\
depth_test_root_path, semantics_train_root_path, semantics_test_root_path, \
num_epochs,batch_size, use_gpu, pretrained, model_weights_path, resume, \
loss_type, learning_rate, read_semantics = sys.argv[1:]

learning_rate = float(learning_rate)
use_gpu = use_gpu.lower() == 'true'
pretrained = pretrained.lower() == 'true'
read_semantics = read_semantics.lower() == 'true'

if resume.lower() == 'none':
    resume = None

if read_semantics:
    net = create_join_net(pretrained=False)

    trainer = JointTrainer(model=net,
                           use_gpu=use_gpu,
                           input_train_root_path=input_train_root_path,
                           depth_train_root_path=depth_train_root_path,
                           semantics_train_root_path=semantics_train_root_path,
                           input_test_root_path=input_test_root_path,
                           depth_test_root_path=depth_test_root_path,
                           semantics_test_root_path=semantics_test_root_path,
                           num_epochs=int(num_epochs),
                           batch_size=int(batch_size),
                           resume=resume,
                           loss_type=loss_type
                           )

    if not pretrained:
        trainer.train_model(checkpoint_freq=1)

    else:
        model_dict = torch.load(model_weights_path)
        net.load_state_dict(model_dict)
        trainer.model = net

        trainer.model.eval()
        av_depth_loss, av_depth_acc, semantic_loss, semantic_acc = trainer.validate()
        print('depth loss: %.4f' % av_depth_loss)
        print('depth acc: %.4f' % av_depth_acc)
        print('semantic loss: %.4f' % semantic_loss)
        print('semantic acc: %.4f' % semantic_acc)

else:
    # creating our network
    net = create_baseline(pretrained=True)

    trainer = DepthTrainer(model=net, use_gpu=use_gpu,
                           input_train_root_path=input_train_root_path,
                           target_train_root_path=depth_train_root_path,
                           input_test_root_path=input_test_root_path,
                           target_test_root_path=depth_test_root_path,
                           num_epochs=int(num_epochs),
                           batch_size=int(batch_size),
                           resume=resume,
                           loss_type=loss_type,
                           learning_rate=learning_rate
                           )

    if not pretrained:
        trainer.train_model(checkpoint_freq=1)

    else:
        model_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(model_dict)
        trainer.model = net
        trainer.model.eval()
        av_loss, av_acc = trainer.run(validate=True)
        print('av loss: ' + str(av_loss) + ' av acc: ' + str(av_acc))







