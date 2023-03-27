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

def getSavedFilePath(
        saved_filename_prefix,
        learning_rate_start,
        learning_rate_step,
        learning_rate_gamma,
        epochs,
        timestamps,
    ):
    """Get the file path of the saved .pt file."""

    return f'saved/{saved_filename_prefix}-{learning_rate_start}-{learning_rate_step}-{learning_rate_gamma}-{epochs}-{timestamps}.pt'
