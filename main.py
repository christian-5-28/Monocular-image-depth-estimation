import sys
import os
os.chdir(os.getcwd())
print(os.getcwd())

from models.DepthModel import DepthModel
from models.DS_net import DsNet
from models.depthModel_regularized import *
from models.trainer import Trainer

from utils.configuration import get_config
from utils.utils import prepare_dirs
from utils.dataset import *

config = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(_):

    # creating the directories for model weights,
    # logs and tensorboard logs
    prepare_dirs(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    read_semantics = config.read_semantics

    if read_semantics and not config.semantic_regularizer:
        model = DsNet(config)
        print("using DS-net architecture!")

    elif config.semantic_regularizer:
        model = DepthModelRegularized(config)
        print("using Depth-net regularized architecture!")

    else:
        model = DepthModel(config)
        print("using depth-net architecture!")

    # creating the dataset and the data loader objects
    dataset = Dataset(config=config, name="kitti", read_semantics=read_semantics)
    data_loader = DataLoader(dataset=dataset, shuffle_data=True)

    # creating our trainer
    trainer = Trainer(model=model,
                      data_loader=data_loader,
                      config=config,
                      rng=rng)

    # training session
    if config.is_train:
        if config.resume_training:
            trainer.resume_training()
        else:
            trainer.train()

    # test session
    else:
        trainer.test(show_figure=False, save_figures=True, is_real_kitti=True)


if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
