import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.autograd import Variable

from datasets.MNIST.MnistDataOrganizer import MnistDataOrganizer
from datasets.WISDOM.WisdomDataset import WisdomDataOrganizer
from datasets.WISDOM.WisdomTimeseriesDataset import WisdomTimeSeriesDataOrganizer
from models.BaselineModel import BaselineModel
from models.LSTMBaselineModel import LSTMBaselineModel
from models.VIBMnist import VIBMnist
from models.VIBWisdom import VIBWisdom
from models.WeightedEMAModel import WeightEmaModel


class AbstractSolver(ABC):

    def __init__(self, config):
        self.args = config['solver']
        self.logger = logging.getLogger(config['log_name'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epoch = self.args['num_epochs']
        self.lr = self.args['learning_rate']
        self.is_testing = self.args['is_testing']
        self.data_loaders = self.get_data_loaders(config['dataset'])

        self.model = self.get_model(config['model'])
        self.weighted_ema_model = WeightEmaModel(self.get_model(config['model']), self.model.state_dict(), decay=0.999)

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

        self.global_iter = 0
        self.global_epoch = 0
        self.history = {"accuracy": 0}
        self.metrics = {}
        self.init_metrics()
        self.tensorboard = self.args['set_tensorboard']
        self.checkpoint_directory = Path(self.args['checkpoint_directory']).joinpath(self.logger.name)
        if not self.is_testing:
            if not self.checkpoint_directory.exists():
                self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
            if self.args['load_checkpoint']:
                self.load_checkpoint()

            # Tensorboard

            if self.tensorboard:
                self.env_name = self.args['model_name']
                self.summary_dir = Path(self.args['summary_directory']).joinpath(self.logger.name)
                if not self.summary_dir.exists():
                    self.summary_dir.mkdir(parents=True, exist_ok=True)
                self.tf = SummaryWriter(log_dir=self.summary_dir)
                self.tf.add_text(tag='argument', text_string=str(config), global_step=self.global_epoch)

    def get_model(self, model_config):
        if self.args['model_name'] in 'baseline':
            return BaselineModel().to(self.device)
        elif self.args['model_name'] in 'VIBMnist':
            return VIBMnist(model_config).to(self.device)
        elif self.args['model_name'] in 'VIBWisdom':
            return VIBWisdom(model_config).to(self.device)
        elif self.args['model_name'] in 'LSTMBaseline':
            return LSTMBaselineModel(model_config).to(self.device)
        return None

    def get_data_loaders(self, dataset_config):
        if self.args['dataset_name'] in 'MNIST':
            return MnistDataOrganizer(dataset_config).get_data_loaders()
        elif self.args['dataset_name'] in 'WISDOM':
            return WisdomDataOrganizer(dataset_config).get_data_loaders()
        elif self.args['dataset_name'] in 'WISDOMTimeSeries':
            return WisdomTimeSeriesDataOrganizer(dataset_config).get_data_loaders()
        return None

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.model.train()
            self.weighted_ema_model.model.train()
        elif mode == 'eval':
            self.model.eval()
            self.weighted_ema_model.model.eval()
        else:
            raise Exception('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        total_num = 0

        for e in range(self.epoch):
            self.global_epoch += 1
            idx = 0
            for idx, (instances, labels) in enumerate(self.data_loaders['train'], start=1):
                self.global_iter += 1

                x = Variable(instances.to(self.device))
                y = Variable(labels.to(self.device))

                total_num += y.size(0)

                model_results = self.get_predictions(x)
                loss = self.get_loss(model_results, y)

                self.set_metrics_values(x, y, model_results, count=True)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.weighted_ema_model.update(self.model.state_dict())

            if self.global_epoch % 1 == 0:
                status = f"Epoch {self.global_epoch} "

                status = self.get_metrics_status(idx, status, total_num)
                self.logger.debug(status)
                if not self.is_testing and self.tensorboard:
                    self.update_tensorboard()
                total_num = 0

            if (self.global_epoch % 2) == 0:
                self.scheduler.step()

            self.init_metrics()
            self.test()
            self.init_metrics()
        self.logger.info(f"{self.logger.name} {self.history}")

    def test(self, save_checkpoint=True):
        self.set_mode('eval')
        total_num = 0
        idx = 0
        for idx, (instances, labels) in enumerate(self.data_loaders['test'], start=1):
            x = Variable(instances.to(self.device))
            y = Variable(labels.to(self.device))

            model_results = self.get_predictions(x, is_test=True)

            self.set_metrics_values(x, y, model_results, count=True)

            total_num += y.size(0)
            _ = self.get_loss(model_results, y)

        status = 'Evaluation '
        status = self.get_metrics_status(idx, status, total_num)
        self.logger.debug(status)

        if not self.is_testing:
            self.update_history(save_checkpoint)

            if self.tensorboard:
                self.update_tensorboard(mode='test')
        self.set_mode('train')

    def get_metrics_status(self, idx, status, total_num):
        for k, v in self.metrics.items():
            if 'accuracy' not in k:
                self.metrics[k] = v / idx
                status += f'{k} :{(v / idx):.4f} '
            else:
                self.metrics[k] = v / total_num
                status += f"{k}: {(v / total_num):.4f} "
                status += f"error {k}: {1 - (v / total_num):.4f} "
        return status

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
            'net': self.model.state_dict(),
            'net_ema': self.weighted_ema_model.model.state_dict(),
        }
        optim_states = {
            'optim': self.optim.state_dict(),
        }
        states = {
            'iter': self.global_iter,
            'epoch': self.global_epoch,
            'history': self.history,
            'args': self.args,
            'model_states': model_states,
            'optim_states': optim_states,
        }

        file_path = self.checkpoint_directory.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        self.logger.debug(f"=> saved checkpoint '{file_path}' (iter {self.global_iter})")

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.checkpoint_directory.joinpath(filename)
        if file_path.is_file():
            self.logger.debug("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.model.load_state_dict(checkpoint['model_states']['net'])
            self.weighted_ema_model.model.load_state_dict(checkpoint['model_states']['net_ema'])

            self.logger.debug("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print(f"=> no checkpoint found at '{file_path}'")

    def get_predictions(self, x, is_test=False):
        return self.weighted_ema_model.model(x) if is_test else self.model(x)

    @abstractmethod
    def get_loss(self, model_results, y):
        pass

    @abstractmethod
    def set_metrics_values(self, x, y, y_predicted, count):
        pass

    @abstractmethod
    def update_tensorboard(self, mode='train'):
        pass

    @abstractmethod
    def update_history(self, save_checkpoint):
        pass

    @abstractmethod
    def init_metrics(self):
        pass


