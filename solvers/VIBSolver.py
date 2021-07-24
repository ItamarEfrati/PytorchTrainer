import math

import torch
import torch.nn.functional as F

from solvers.AbstractSolver import AbstractSolver


class VIBSolver(AbstractSolver):

    def __init__(self, config):
        super().__init__(config)
        self.beta = self.args['beta']
        self.number_of_samples = self.args['number_of_samples']
        self.history['avg_accuracy'] = 0

    def init_metrics(self):
        self.metrics = {
            "izy_bound": 0,
            "izx_bound": 0,
            "accuracy": 0,
            "avg_accuracy": 0,
            "class_loss": 0,
            "info_loss": 0,
            "total_loss": 0
        }

    def get_loss(self, model_results, y):
        (mu, std), logit = model_results

        class_loss = F.cross_entropy(logit, y).div(math.log(2))
        info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        total_loss = class_loss + self.beta * info_loss
        label_entropy = math.log(self.args['number_of_classes'], 2)

        self.metrics['izy_bound'] += (label_entropy - class_loss).item()
        self.metrics['izx_bound'] += info_loss.item()
        self.metrics['total_loss'] += total_loss.item()
        self.metrics['class_loss'] += class_loss.item()
        self.metrics['info_loss'] += info_loss.item()

        return total_loss

    def set_metrics_values(self, x, y, model_results, count):
        (_, _), logit = model_results
        prediction = F.softmax(logit, dim=1).max(1)[1]

        self.metrics['accuracy'] += torch.eq(prediction, y).sum().item() if count else torch.eq(prediction,
                                                                                                y).float().mean().item()

        if self.number_of_samples:
            _, avg_soft_logit = self.model(x, self.number_of_samples)
            avg_prediction = avg_soft_logit.max(1)[1]
            self.metrics['avg_accuracy'] += torch.eq(avg_prediction, y).sum().item() if count else torch.eq(
                avg_prediction, y).float().mean().item()

    def update_tensorboard(self, mode='train'):
        self.tf.add_scalars(main_tag='performance/accuracy',
                            tag_scalar_dict={
                                f'{mode}_one-shot': self.metrics['accuracy'],
                                f'{mode}_multi-shot': self.metrics['avg_accuracy']},
                            global_step=self.global_iter)
        self.tf.add_scalars(main_tag='performance/error',
                            tag_scalar_dict={
                                f'{mode}_one-shot': 1 - self.metrics['accuracy'],
                                f'{mode}_multi-shot': 1 - self.metrics['avg_accuracy']},
                            global_step=self.global_iter)
        self.tf.add_scalars(main_tag='performance/cost',
                            tag_scalar_dict={
                                f'{mode}_one-shot_class': self.metrics['class_loss'],
                                f'{mode}_one-shot_info': self.metrics['info_loss'],
                                f'{mode}_one-shot_total': self.metrics['total_loss']},
                            global_step=self.global_iter)
        self.tf.add_scalars(main_tag=f'mutual_information/{mode}',
                            tag_scalar_dict={
                                'I(Z;Y)': self.metrics['izy_bound'],
                                'I(Z;X)': self.metrics['izx_bound']},
                            global_step=self.global_iter)

    def update_history(self, save_checkpoint):
        if self.history['avg_accuracy'] < self.metrics['avg_accuracy']:
            for k, v in self.metrics.items():
                self.history[k] = self.metrics[k]
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if save_checkpoint:
                self.save_checkpoint('best_acc.tar')
