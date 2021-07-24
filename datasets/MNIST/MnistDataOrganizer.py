import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from datasets.AbstractDataset import DataOrganizer


class MnistDataOrganizer(DataOrganizer):

    def __init__(self, config):
        super().__init__(config)
        self.root = "datasets"
        self.transformers = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,)), ])

    def get_data(self):
        return MNIST(root=self.root, train=True, transform=self.transformers, download=True), \
               MNIST(root=self.root, train=False, transform=self.transformers, download=False)
