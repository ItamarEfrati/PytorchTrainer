from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class DataOrganizer(ABC):
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']

    def get_data_loaders(self):
        data_loaders = {}
        train_data, test_data = self.get_data()
        data_loaders['train'] = DataLoader(train_data,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           drop_last=True)

        data_loaders['test'] = DataLoader(test_data,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=1,
                                          drop_last=False)
        return data_loaders

    @abstractmethod
    def get_data(self):
        pass
