import torch

import torch.nn as nn
import torch.nn.functional as F

from solvers.AbstractSolver import AbstractSolver


class BaselineSolver(AbstractSolver):

    def __init__(self, config):
        super(BaselineSolver, self).__init__(config)
        self.criterion = nn.CrossEntropyLoss()

    def init_metrics(self):
        self.metrics = {
            "accuracy": 0,
            "total_loss": 0
        }

    def update_history(self, save_checkpoint):
        if self.history['accuracy'] < self.metrics['accuracy']:
            self.history['accuracy'] = self.metrics['accuracy']
            self.history['total_loss'] = self.metrics['total_loss']
            self.history['epoch'] = self.global_epoch
            self.history['iteration'] = self.global_iter
            if save_checkpoint:
                self.save_checkpoint('best_acc.tar')

    def set_metrics_values(self, x, y, model_results, count):
        logits = model_results
        y_predicted = F.softmax(logits, dim=1).max(1)[1]
        correct = torch.eq(y_predicted, y).sum() if count else torch.eq(y_predicted, y).float().mean()
        self.metrics['accuracy'] += correct.item()

    def update_tensorboard(self, mode='train'):
        self.tf.add_scalars(main_tag='performance/accuracy',
                            tag_scalar_dict={f'{mode}_one-shot': self.metrics['accuracy']},
                            global_step=self.global_iter)
        self.tf.add_scalars(main_tag='performance/error',
                            tag_scalar_dict={f'{mode}_one-shot': 1 - self.metrics['accuracy']},
                            global_step=self.global_iter)

    def get_loss(self, model_results, y):
        total_loss = self.criterion(model_results, y)
        self.metrics['total_loss'] += total_loss.item()
        return total_loss
