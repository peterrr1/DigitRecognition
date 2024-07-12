from torch import nn
import torch
from pipeline import LoadDataset
from tqdm import tqdm
import numpy as np
import csv
from torch.utils.data import DataLoader, random_split
from typing import List, Callable, Optional
import visualize


class Model(nn.Module):
    """
    A convolutional neural network model for image classification.

    ...

    Attributes
    ----------
    features : nn.Sequential
        a sequential container of modules representing the feature extraction part of the model
    classifier : nn.Sequential
        a sequential container of modules representing the classification part of the model

    Methods
    -------
    forward(x: torch.Tensor):
        Defines the computation performed at every call.
    """
    def __init__(self, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        _in_features = 512
        _out_features = 10

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        features: List[nn.Module] = [
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = 32,
                    kernel_size = 3),
                norm_layer(32),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = 32,
                    out_channels = 64,
                    kernel_size = 3),
                norm_layer(64),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = 64,
                    out_channels = 128,
                    kernel_size = 3),
                norm_layer(128),
                nn.MaxPool2d(kernel_size = 2, stride = 1),
                nn.ReLU(),    
        ]

        

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = _in_features, out_features = 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = _out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x



def train(train_ds, val_ds, model, epochs):
    model.train(True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)

    softmax = nn.LogSoftmax(dim = 1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in (pbar_e := tqdm(range(epochs))):
        pbar_e.set_description(f'Epoch [{epoch + 1}/{epochs}]')

        for i, (images, labels) in enumerate(pbar_t := tqdm(train_ds)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            outputs = softmax(outputs)

            if i % 25 == 0:
                pbar_t.set_description(f'TRAINING - Batch [{i + 1}/{len(train_ds)}], Loss: {loss.item()}, Accuracy: {(outputs.argmax(1) == labels).float().mean().item()}')
        validate(val_ds, model)


def validate(val_ds, model):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.LogSoftmax(dim = 1)
    avg_loss, avg_acc = 0, 0

    with torch.no_grad():
        for _, (images, labels) in enumerate(val_ds):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = softmax(outputs)

            result = (outputs.argmax(1) == labels).float().mean().item()

            avg_acc += result
            avg_loss += loss

    print(f'VALIDATION - Validation loss: {avg_loss/len(val_ds)}, Validation accuracy: {avg_acc/len(val_ds)}')
    
def test(test_ds, model):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.LogSoftmax(dim = 1)
    avg_loss, avg_acc = 0, 0

    y_pred = np.array([])
    y_true = np.array([])

    with torch.no_grad():
        for _, (images, labels) in enumerate(test_ds):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = softmax(outputs)

            y_pred = np.append(y_pred, outputs.argmax(1).squeeze().tolist())
            y_true = np.append(y_true, labels.squeeze().tolist())

            result = (outputs.argmax(1) == labels).float().mean().item()

            avg_loss += loss
            avg_acc += result

    
    print(f'TEST - Test loss: {avg_loss/len(test_ds)}, Test accuracy: {avg_acc/len(test_ds)}')

    return y_pred, y_true


def predict(model, test_ds, filename):
    path = f'./{filename}'
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f'Creating {filename}!')

    with open(path, 'w') as file:
        file = csv.writer(file, delimiter=',')
        file.writerow(['ImageId', 'Label'])
    
    with open(path, 'a') as file:
        file = csv.writer(file, delimiter=',')
        with torch.no_grad():
            for i, (image, _) in enumerate(test_ds):
                image = image.to(device)

                prediction = model(image)

                file.writerow([i + 1, prediction.argmax(1).item()])



def main():
    
    dataset = LoadDataset('data/train.csv', normalize = True)
    submis_ds = LoadDataset('data/test.csv', normalize = True, is_test_ds = True)

    train_data, val_data, test_data = random_split(dataset, [0.7, 0.2, 0.1], generator = torch.Generator().manual_seed(100))

    train_data = DataLoader(train_data, batch_size = 64, shuffle = True)
    val_data = DataLoader(val_data, batch_size = 64, shuffle = True)
    test_data = DataLoader(test_data, batch_size = 64, shuffle= True)

    submis_ds = DataLoader(submis_ds, batch_size = 1, shuffle = False)
    
    model = Model()

    train(train_data, val_data, model, 5)
    y_pred, y_true = test(test_data, model)

    visualize.create_barplot(dataset)
    visualize.show_images(dataset)
    visualize.create_confusion_matrix(y_true, y_pred, labels = np.linspace(0, 9, 10))

    predict(model, submis_ds, 'submission.csv')

if __name__ == '__main__':
    main()