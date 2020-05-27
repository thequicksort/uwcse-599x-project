import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import matplotlib.pyplot as plot

import logging

NANOPORTER_RAW_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy"
NANOPORTER_CLASSES_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy"
NANOPORTER_DIR = "/Users/dna/Developer/590_project/"

raw_data_path = Path(NANOPORTER_DIR, NANOPORTER_RAW_FILENAME).absolute()
classes_path = Path(NANOPORTER_DIR, NANOPORTER_CLASSES_FILENAME).absolute()


download_instructions = """
rsync -zaP jdunstan@misl-a.cs.washington.edu:/disk1/pore_data/NanoporeTERs/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy .
rsync -zaP jdunstan@misl-a.cs.washington.edu:/disk1/pore_data/NanoporeTERs/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy .
    """

def get_raw_data_and_labels(raw_data_path, classes_path):
    try:
        raw = np.load(raw_data_path)
        classes = np.load(classes_path)
        signal_and_classes = np.array(zip(raw, classes))
        return signal_and_classes
    except Exception:
        logging.error("Unable to find data at path")

def shuffled_data(data, seed=0xBEEF5EED):
    np.random.seed(seed=seed)
    shuffled_indecies = np.arange(0, len(data), 1)
    np.random.shuffle(shuffled_indecies)
    data = np.array(data)[shuffled_indecies]
    return data

def split_data(raw_data_path, classes_path):
    # Normally I'd use sklearn shuffle to handle this,
    # but I'd prefer to minimize the dependencies in this project
    # for easier cross-platform porting.
    raw_and_labels = get_raw_data_and_labels(raw_data_path, classes_path)
    labeled_data = shuffled_data(raw_and_labels)
    n = len(labeled_data)
    train_percentage = 0.9
    validation_percentage = 0.05

    test_percentage = 1.0 - train_percentage - validation_percentage

    n_train = int(train_percentage * n)
    n_val = int(validation_percentage * n)
    n_train = n_val

    X_train = labeled_data[0: n_train]
    X_val = labeled_data[n_train: n_train + n_val]
    X_test = labeled_data[n_train + n_val: -1]
    print(f"n_train: {n_train}\tn_val: {n_val}\tn_test: {n_test}")

def train_epochs(epochs=1000, print_every=10):
    assert print_every > 1, "Print every must be greater than 1"
    for i in range(n):
        epoch_loss = 0.343
        if i % print_every == print_every - 1:
            print(f"epoch: {i}\tloss:{epoch_loss:.6f}")
    assert test_percentage > 0.0, "Must have some amount of test data"

class NanoporeterClassifierLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=False, nonlinearity="tanh")
        self.fully_connected = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out_0, (hidden_1, cell_1) = self.lstm(x, (hidden_0, cell_0))

        out = self.fully_connected(out_0)
        softmaxed_out = F.log_softmax(out)
        return softmaxed_out


