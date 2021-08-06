from typing import Any

import torch.nn.functional as F

from torch import nn

from utils.ModelsUtils import xavier_init


class LSTMBaselineModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, model_config):
        super(LSTMBaselineModel, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(3, 6, batch_first=True)
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 12)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(12, model_config['target_size'])

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        o = F.relu(self.fc1(hn.squeeze()))
        o = F.relu(self.fc2(o))
        tag_space = self.hidden2tag(o)
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores

    def init_weights(self):
        xavier_init(self._modules)
