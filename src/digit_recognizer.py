import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, shift
from sklearn.model_selection import train_test_split
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from src.common import data_augmentation_rotate, data_augmentation_shift, test_model, train_and_validate_model

BATCH_SIZE = 4096
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
L2_PENALTY = 0


class DigitRecognizerNN(nn.Module):

    def __init__(self):
        super(DigitRecognizerNN, self).__init__()
        self.min_loss = 1e10
        self.max_state_dict = None

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(256 * 5 * 5, 256)
        self.drop3 = nn.Dropout()
        self.fc2 = nn.Linear(256, 64)
        self.drop4 = nn.Dropout()
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop2(self.relu2(self.conv2(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.drop3(self.fc1(x))
        x = self.drop4(self.fc2(x))
        return self.softmax(self.fc3(x))

    def record_state(self, loss):
        # used for early stopping
        self.min_loss = loss
        self.max_state_dict = self.state_dict()


def load_training_data_from_csv(filename):
    dataset = pd.read_csv(filename, dtype=np.float32)
    labels = dataset.label.values
    features = dataset.loc[:, dataset.columns != 'label'].values / 255.

    data_train, data_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.25)
    all_features = data_train.reshape(-1, 28, 28)
    all_labels = labels_train.reshape(-1)
    for trans in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        data_shifted, labels_shifted = data_augmentation_shift(data_train, labels_train, shift, shape=(28, 28), shift=trans)
        all_features = np.r_[all_features, data_shifted]
        all_labels = np.r_[all_labels, labels_shifted]

    data_shifted, labels_shifted = data_augmentation_rotate(data_train, labels_train, rotate, shape=(28, 28), angle=10)
    all_features = np.r_[all_features, data_shifted]
    all_labels = np.r_[all_labels, labels_shifted]

    data_train = torch.from_numpy(all_features.reshape(-1, 1, 28, 28)).type(torch.cuda.FloatTensor)
    labels_train = torch.from_numpy(all_labels.reshape(-1)).type(torch.cuda.LongTensor)

    data_val = torch.from_numpy(data_val.reshape(-1, 1, 28, 28)).type(torch.cuda.FloatTensor)
    labels_val = torch.from_numpy(labels_val.reshape(-1)).type(torch.cuda.LongTensor)

    train = torch.utils.data.TensorDataset(data_train, labels_train)
    val = torch.utils.data.TensorDataset(data_val, labels_val)
    return train, val


def load_test_data_from_csv(filename):
    dataset = pd.read_csv(filename, dtype=np.float32)
    data_test = dataset.loc[:, dataset.columns != 'label'].values / 255.
    data_test = torch.from_numpy(data_test.reshape(-1, 1, 28, 28)).type(torch.cuda.FloatTensor)

    testing = torch.utils.data.TensorDataset(data_test)
    return testing


def digit_recognizer_training_main():
    # batch_size, epoch and iteration
    train, val = load_training_data_from_csv('data/DigitRecognizer/train.csv')
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    print('data loaded')

    model = DigitRecognizerNN()
    model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_PENALTY)

    history = train_and_validate_model(model, optimizer, loss, train_loader, val_loader, NUM_EPOCHS, 1)
    plt.plot(list(range(NUM_EPOCHS)), history['loss'])
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('model/DigitRecognizer/loss.eps')

    plt.clf()
    plt.plot(list(range(NUM_EPOCHS)), history['accu'])
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('model/DigitRecognizer/accu.eps')

    torch.save(model.max_state_dict, 'model/DigitRecognizer/model.pt')


def digit_recognizer_testing_main():
    testing = load_test_data_from_csv('data/DigitRecognizer/test.csv')
    testing_loader = torch.utils.data.DataLoader(testing, batch_size=BATCH_SIZE, shuffle=False)

    model = DigitRecognizerNN()
    model.cuda()
    model.load_state_dict(torch.load('model/DigitRecognizer/model.pt'))
    output = test_model(model, testing_loader)

    df = pd.read_csv('data/DigitRecognizer/sample_submission.csv').assign(Label=output)
    df.to_csv('data/DigitRecognizer/submission.csv', index=False)


if __name__ == '__main__':
    if '--train' in sys.argv:
        digit_recognizer_training_main()
    elif '--test' in sys.argv:
        digit_recognizer_testing_main()
