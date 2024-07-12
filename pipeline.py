from torchvision.transforms import v2
from typing import Callable, Optional
import torch
import pandas as pd
from torch.nn.functional import one_hot

class LoadDataset:
    """
    A class used to load and preprocess a dataset from a CSV file.

    ...

    Attributes
    ----------
    path : str
        a string representing the path to the CSV file
    normalize : bool
        a boolean indicating whether to normalize the data (default is False)
    is_test_ds : bool
        a boolean indicating whether the dataset is a test dataset (default is False)
    transform : torchvision.transforms.Compose
        a torchvision Compose object used to normalize the data
    data : torch.Tensor
        a tensor representing the data loaded from the CSV file
    label : torch.Tensor
        a tensor representing the labels loaded from the CSV file

    Methods
    -------
    __len__():
        Returns the length of the data.
    __getitem__(idx: int):
        Returns the data and label at the given index.
    load_dataset(path: str, transform: Optional[Callable[..., v2.Compose]] = None):
        Loads the dataset from the given path and applies the given transform.
    """
    def __init__(self, path: str, normalize: bool = False, is_test_ds: bool = False) -> None:
        print(f'Loading dataset from {path}')
        self.path = path
        self.normalize = normalize
        self.is_test_ds = is_test_ds

        # Define the transformation to normalize the data
        self.transform = v2.Compose([
            v2.Normalize([0.5, ], [0.5, ])
        ])

        # Load the dataset
        self.load_dataset(self.path)

    # Define the length of the dataset as the length of the data
    def __len__(self) -> int:
        return len(self.data)

    # Define how to get an item from the dataset
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.label[idx]

    # Define how to load the dataset
    def load_dataset(self, path: str, transform: Optional[Callable[..., v2.Compose]] = None) -> None:
        df = pd.read_csv(path)

        # If it's not a test dataset, separate the labels and the data
        if not self.is_test_ds:
            self.label = torch.tensor(df.iloc[:, 0].values)
            self.data = torch.tensor(df.iloc[:, 1:].values).float().reshape(-1, 1, 28, 28) 
        else:
            # If it's a test dataset, only load the data
            self.label = torch.tensor(df.iloc[:, 0].values)
            self.data = torch.tensor(df.values).float().reshape(-1, 1, 28, 28) 

        # If normalize is True, normalize the data
        if self.normalize:
            self.data = self.transform(self.data / torch.max(self.data))   
        
    