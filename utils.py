# PyTorch imports
import torchvision

class MNIST(torchvision.datasets.MNIST):

    cache = {}

    def __init__(self, path, train=True, transform=None):

        super().__init__(path, download=True, train=train, transform=transform)

        # Access all samples to pre-cache them.
        for idx in range(self.__len__()):
            self.__getitem__(idx)

    def __len__(self):

        return super().__len__()
    
    def __getitem__(self, idx):

        if idx in self.cache:
            return self.cache[idx]
        else:
            self.cache[idx] = super().__getitem__(idx)
            return self.cache[idx]
        
class FashionMNIST(torchvision.datasets.FashionMNIST):

    cache = {}

    def __init__(self, path, train=True, transform=None):

        super().__init__(path, download=True, train=train, transform=transform)

        # Access all samples to pre-cache them.
        for idx in range(self.__len__()):
            self.__getitem__(idx)

    def __len__(self):

        return super().__len__()
    
    def __getitem__(self, idx):

        if idx in self.cache:
            return self.cache[idx]
        else:
            self.cache[idx] = super().__getitem__(idx)
            return self.cache[idx]
