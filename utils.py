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

def getSavedFilePath(saved_filename_prefix, learning_rate, use_learning_rate_decay):
    """Get the file path of the saved .pt file."""

    # Whether add -decay to the filename.
    add_decay = lambda use_learning_rate_decay: '-decay' if use_learning_rate_decay else ''

    return f'saved/{saved_filename_prefix}-{learning_rate}{add_decay(use_learning_rate_decay)}.pt'
