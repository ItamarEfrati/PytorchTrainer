import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from numbers import Number

from utils.ModelsUtils import xavier_init


class VIBMnist(nn.Module):
    def __init__(self, config):
        super(VIBMnist, self).__init__()
        self.K = config['encoding_size']

        self.encode = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2 * self.K))

        self.decode = nn.Sequential(
            nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        statistics = self.encode(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)

        if num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_())

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])