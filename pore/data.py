import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import os

NANOPORTER_RAW_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy"
NANOPORTER_CLASSES_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy"
HOME = os.environ["HOME"]
NANOPORTER_DIR = Path(HOME, "Developer/misl/uwcse-599x-project/pore/data")

raw_data_path = Path(NANOPORTER_DIR, NANOPORTER_RAW_FILENAME).absolute()
classes_path = Path(NANOPORTER_DIR, NANOPORTER_CLASSES_FILENAME).absolute()

class NanoporeTERDataset(Dataset):

    def __init__(self, raw_file, labeled_file, transform=None):
        self.raw_data = np.load(Path(raw_file).absolute())
        self.labels = np.load(Path(labeled_file).absolute())
        self.transform = transform

    def __len__(self):
        length = len(self.raw_data)
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal = self.raw_data[idx]
        label = self.labels[idx]
        item = {"signal": signal, "label": label}
        if self.transform:
            item = self.transform(item)
        return item


def split_dataset(dataset, train_percent=0.9, validation_percent=0.05, test_percent=0.05, seed=None):
    length = len(dataset)
    distribution_sum = train_percent + validation_percent + test_percent
    if not math.isclose(distribution_sum, 1.0):
        error_message = f"train/val/test split percentages ({train_percent}, {validation_percent}, {test_percent}) do not add up to 1.0"
        raise Exception(error_message)

    train_length = int(train_percent * length)
    val_length = int(validation_percent * length)
    test_length = int(test_percent * length)

    # if data was left out due to rounding, add the remaining data to the training set
    remaining = length - ( train_length + val_length + test_length )
    if remaining > 0:
        train_length += remaining

    print(f"train: {train_length}\tval: {val_length}\ttest: {test_length}\ttotal: {length}")
    if seed:
        torch.random.manual_seed(seed)
    train, val, test = torch.utils.data.random_split(dataset, [train_length, val_length, test_length])
    return train, val, test




def get_data_loader(data_dir=NANOPORTER_DIR, raw_filename=NANOPORTER_RAW_FILENAME, labeled_filename=NANOPORTER_CLASSES_FILENAME, batch_size=4, shuffle=True, num_workers=4):
    dataset = NanoporeTERDataset(data_dir, raw_filename, labeled_filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    #fig = plt.figure()
    raw_data_path = Path(NANOPORTER_DIR, NANOPORTER_RAW_FILENAME).absolute()
    classes_path = Path(NANOPORTER_DIR, NANOPORTER_CLASSES_FILENAME).absolute()

    dataset = NanoporeTERDataset(NANOPORTER_DIR, NANOPORTER_RAW_FILENAME, NANOPORTER_CLASSES_FILENAME)
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(75, 6))
    for row in range(rows):
        for col in range(cols):
            sample_n = np.random.randint(0, len(dataset))
            sample = dataset[sample_n]

            ax = axes[row, col]
            ax.plot(sample["signal"])
            ax.set_ylim(bottom=0, top=1)
            title = "{0} ({1})".format(sample["label"], sample_n)
            ax.set_title(title)
    plt.show()