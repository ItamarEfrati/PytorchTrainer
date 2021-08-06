import joblib
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch import tensor
from torch.utils.data import TensorDataset

from datasets.AbstractDataset import DataOrganizer


class WisdomTimeSeriesDataOrganizer(DataOrganizer):

    def __init__(self, config):
        super().__init__(config)
        self.data_array = joblib.load("datasets/WISDOM/raw/data_10sec_20Hz_compress_3.gz")
        self.labels_array = joblib.load("datasets/WISDOM/raw/label_10sec_20Hz")
        self.scaler = MinMaxScaler()

    def get_train_test_indices(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
        return next(sss.split(self.data_array, self.labels_array))

    def get_data(self):
        train_indices, test_indices = self.get_train_test_indices()
        train_x, test_x = self.data_array[train_indices], self.data_array[test_indices]
        train_x, test_x = train_x.reshape(train_x.shape[0], 200, 3), test_x.reshape(test_x.shape[0], 200, 3)
        train_y, test_y = self.labels_array[train_indices].reshape(-1), self.labels_array[test_indices].reshape(-1)

        temp_train_x = self.scaler.fit_transform(train_x.reshape(train_x.shape[1], -1).T)
        train_x = temp_train_x.T.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2])
        temp_test_x = self.scaler.transform(test_x.reshape(test_x.shape[1], -1).T)
        test_x = temp_test_x.T.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2])

        return TensorDataset(tensor(train_x, dtype=torch.float), tensor(train_y, dtype=torch.long)), \
               TensorDataset(tensor(test_x, dtype=torch.float), tensor(test_y, dtype=torch.long))
