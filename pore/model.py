import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from torch.autograd import Variable
import argparse

import torch.autograd as autograd
import time

#from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot

import logging
import os
import csv
import datetime

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pore.data import NanoporeTERDataset, split_dataset

from torch.utils.data import DataLoader


DEFAULT_SEED = 0xDEAD5EED

download_instructions = """
rsync -zaP jdunstan@misl-a.cs.washington.edu:/disk1/pore_data/NanoporeTERs/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy .
rsync -zaP jdunstan@misl-a.cs.washington.edu:/disk1/pore_data/NanoporeTERs/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy .
    """


class NanoporeterClassifierLSTM(nn.Module):

    def __init__(self, signal_length, hidden_size, batch_size, device, output_size=1, num_layers=2):
        super(NanoporeterClassifierLSTM, self).__init__()
        self.signal_length = signal_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device

        self.to(self.device)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # input_size == 1, we're only looking at one feature: electropheoretic current
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False).to(self.device)

        self.fully_connected_1 = nn.Linear(self.hidden_size, output_size).to(self.device)
        #self.fully_connected_2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2).to(self.device)
        #self.fully_connected_3 = nn.Linear(self.hidden_size // 2, output_size).to(self.device)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden_0 = Variable(torch.zeros(self.num_layers, self.signal_length, self.hidden_size).double())
        cell_0 = Variable(torch.zeros(self.num_layers, self.signal_length, self.hidden_size).double())
        return (hidden_0, cell_0)

    def forward(self, x):
        #hidden_0 = Variable(torch.zeros(self.num_layers, self.signal_length, self.hidden_size)).to(self.device)
        #cell_0 = Variable(torch.zeros(self.num_layers, self.signal_length, self.hidden_size)).to(self.device)
        out, (ht, ct) = self.lstm(x.float())#, (hidden_0, cell_0))
        output = self.fully_connected_1(out[:, -1, :].data)
        output = self.sigmoid(output)
        return output
#         out = self.relu(self.fully_connected_2(out))
#         out = self.relu(self.fully_connected_3(out))
        # softmaxed_out = F.softmax(out, dim=1)
        # return softmaxed_out
        #out = out.view(-1)
        return out


def train_epoch(model, training_dataloader, device, loss_function, optimizer=None, epoch=0, print_epoch=False, print_every=100):
    running_loss = 0.0
    model.zero_grad()

    for i, data in enumerate(training_dataloader):
        #toc = time.time()
        with torch.set_grad_enabled(True):
            signal = data["signal"]
            label = data["label"].squeeze(1)
            signal, label = signal.to(device), label.to(device)

            if optimizer:
                optimizer.zero_grad()

            signal.unsqueeze_(-1)
            estimate = model(signal)
            loss = loss_function(estimate, label)

            # Update metavariables
            running_loss += loss.data

            # Backwards step
            loss.backward()
            if optimizer:
                optimizer.step()

        if print_epoch and (i % print_every == print_every - 1):
            message = f"\tEpoch[{epoch}] [{i}] - Average Loss: {running_loss/(i+1)}"
            print(message)

# Was used for profiling (found GPU on cloud to be 10x faster than my own machine. wow)
#         tic = time.time()
#         diff = tic - toc # on the clock
#         print(f"Single sample time: {diff:.4f}")

    avg_loss = running_loss / len(training_dataloader)
    if print_epoch:
        message = f"Epoch[{epoch}] - Average Loss: {avg_loss}"
        print(message)
    return avg_loss


def validate_epoch(model, validation_dataloader, device, epoch=0, print_epoch=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_dataloader:
            signal = data["signal"]
            labels = data["label"].squeeze(1)
            signal, labels = signal.to(device), labels.to(device)

            output = model(signal)
            certainty, predicted = torch.max(output.data, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()
    accuracy = (correct/total) * 100
    if print_epoch:
        msg = f"Epoch[{epoch}] - Val Accuracy: {accuracy:.3f}%"
        print(msg)
    return accuracy

def train_epochs(model: nn.Module, train_dataset, val_dataset, device, learning_rate=0.05, epochs=10, batch_size=1, shuffle=True, num_workers=4, print_every=100, seed=DEFAULT_SEED):

    #torch.backends.cudnn.benchmark = True

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #Y_train = Y_train.unsqueeze(dim=0) # https://stackoverflow.com/questions/53936136/pytorch-inputs-for-nn-crossentropyloss
    if seed:
        print(f"Using seed in train_epochs: {seed}")
        torch.manual_seed(seed)

    losses = []
    accuracies = []
    epoch_list = []

    for epoch in range(epochs):
        print_epoch = print_every > 0
        loss = train_epoch(model, training_dataloader, device, loss_function, optimizer, epoch=epoch, print_epoch=print_epoch, print_every=print_every)
        accuracy = validate_epoch(model, val_dataloader, device, epoch=epoch, print_epoch=print_epoch)

        losses.append(loss)
        accuracies.append(accuracy)
        epoch_list.append(epoch)

    return zip(epoch_list, losses, accuracies)


def train(raw_file, label_file, device, model_dir=".", seed=DEFAULT_SEED, epochs=10, n_hidden=2, learning_rate=0.05, batch_size=4, shuffle=True, num_workers=4, print_every=100):
    dataset = NanoporeTERDataset(raw_file, label_file)

    train, val, test = split_dataset(dataset, train_percent=0.9, validation_percent=0.05, test_percent=0.05, seed=seed)

    input_size = len(train.dataset.raw_data[0])

    classes = np.unique(train.dataset.labels)
    n_classes = len(classes)
    output_size = n_classes #len(Y_train[0])
    model = NanoporeterClassifierLSTM(input_size, n_hidden, batch_size, device, output_size=output_size)
    #model.hidden = model.init_hidden()

    epochLossAndAccuracy = train_epochs(model, train, val, device, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, num_workers=num_workers, print_every=print_every, seed=seed)
    if model_dir:
        save_model(model, model_dir, "LSTM", epochLossAndAccuracy)


def save_model(model, file_dir, file_prefix, epochLossAndAccuracy):
    date = datetime.date.today().isoformat()
    base_filename = f"model_{file_prefix}_{date}"
    model_filename = f"{base_filename}.pt"
    model_filepath = Path(file_dir, model_filename)
    torch.save(model, model_filepath)
    print(f"Model saved to: {model_filepath}")

    summary_filename = f"{base_filename}_summary.csv"
    summary_filepath = Path(file_dir, summary_filename)
    with open(summary_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy"])
        writer.writerows(epochLossAndAccuracy)
    print(f"Model summary to: {summary_filepath}")

#model = NanoporeterClassifierLSTM(input_size, hidden_size, batch_size, output_size=output_size)



def get_args(passed_args=None):
    # Defaults:
    SEED = 0xBEEF5EED
    MODEL_DIR = path.dirname(path.dirname(path.abspath(__file__))) #Path(HOME, "Developer/misl/uwcse-599x-project/pore/models")

    learning_rate = 1e-3
    num_workers = 6
    batch_size = 1
    epochs = 10
    n_hidden = 2

    parser = argparse.ArgumentParser(description="Training for nanoporeter classifiers")
    parser.add_argument("--raw", action="store", required=True, help="Raw signal data, corresponding to labels data")
    parser.add_argument("--labeled", action="store", required=True, help="Corresponding to labeled classes for raw signal data")
    parser.add_argument("--n_workers", action="store", required=False,type=int, default=num_workers, help="Number of workers to use")
    parser.add_argument("--seed", action="store", required=False, type=lambda x: int(x,0) ,help="Seed to use in torch random number generator") # Using lambda x: int(x,0) to handle hex: https://stackoverflow.com/questions/25513043/python-argparse-fails-to-parse-hex-formatting-to-int-type
    parser.add_argument("--learningrate", "--lr", "-lr", action="store", type=float,required=False, default=learning_rate, help="Training learning rate")
    parser.add_argument("--batchsize", "-b", action="store", required=False,type=int, default=batch_size, help="Worker batch size for data")
    parser.add_argument("--epochs", "-e", action="store", required=False, type=int, default=epochs, help="Number of epochs to train for")
    parser.add_argument("--modeldir", action="store", required=False, default=MODEL_DIR, help="Where to store models")
    parser.add_argument("--n_hidden", action="store", required=False, type=int, default=n_hidden, help="Number of hidden layers")

    args = parser.parse_args(args=passed_args)

    return args
import sys
def main(args):

    # NANOPORTER_RAW_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy"
    # NANOPORTER_CLASSES_FILENAME = "Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy"
    # HOME = os.environ["HOME"]
    # NANOPORTER_DIR = Path(HOME, "Developer/misl/uwcse-599x-project/pore/data")
    # SEED = 0xBEEF5EED

    # root_dir = NANOPORTER_DIR
    # raw_file = NANOPORTER_RAW_FILENAME
    # label_file = NANOPORTER_CLASSES_FILENAME

    # learning_rate = 1e-3
    # num_workers = 6
    # batch_size = 1
    # epochs = 1

    print(f"Training with args: {args.__dict__!s}")
    # Set up Cuda if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train(args.raw, args.labeled, device, model_dir=args.modeldir, seed=args.seed, learning_rate=args.learningrate, n_hidden=args.n_hidden, num_workers=args.n_workers, batch_size=args.batchsize, epochs=args.epochs)


if __name__ == "__main__":
    default_args = """
    --raw=./pore/data/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019.npy
    --labeled=./pore/data/Y00_Y08_MinIONNoise_LBNoise_EcoliNoise_raw20000_03132019_classes.npy
    --epochs=2
    --modeldir=./pore/models/
    --batchsize=40
    --n_workers=5
    --learningrate=0.1
    --n_hidden=1
    --seed=0x5ADBEEF
    """
    default_args = " ".join(default_args.split()).split() # Splits each arg into its own list item.

    args = get_args(passed_args=default_args)
    main(args)


# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
# https://www.curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel