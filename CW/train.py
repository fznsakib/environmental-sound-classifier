#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import sys
import os
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torchvision.transforms import Compose
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data.dataset import UrbanSound8KDataset
# from torchsummary import summary

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

default_dataset_dir = os.getcwd() + '/data/'
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--sgd-momentum", default=0.9, type=float)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--mode", default='LMC', type=str)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):

    if (args.mode not in ['LMC', 'MC', 'MLMC']):
        print('modes allowed: LMC, MC, MLMC')
        exit()

    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset("data/UrbanSound8K_train.pkl", args.mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset("data/UrbanSound8K_test.pkl", args.mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=85, width=41, channels=1, class_count=10, mode=args.mode, dropout=args.dropout)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, mode: str, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout2d(dropout)

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels, # Checkout input channel count
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv1)
        self.batchNorm1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv2)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv3)
        self.batchNorm3 = nn.BatchNorm2d(64)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv4)
        self.batchNorm4 = nn.BatchNorm2d(64)

        ## First fully connected layer
        self.fc1 = nn.Linear(15488, 1024, bias=False) # 15488 = 11 * 22 * 64
        
        # Shape of tensor output from 4th layer will be different due to difference in
        # input dimensions for MLMC. So number of input features to FC1 will be different.
        if mode == 'MLMC':
            self.fc1 = nn.Linear(26048, 1024, bias=False)
  
        self.initialise_layer(self.fc1)

        ## Second fully connected layer
        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # First convolutional layer pass
        x = self.conv1(images)
        x = self.batchNorm1(x)
        x = F.relu(x)

        # Second convolutional layer pass
        # Dropout must be applied to layer and stored, instead of
        # passing as input to next convolutional layer
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third convolutional layer pass
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        # Fourth convolutional layer pass
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ## (batch_size, 4096)
        x = torch.flatten(x, start_dim=1, end_dim=3)

        # First fully connected layer pass
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        ## TASK 6-2: Pass x through the last fully connected layer
        # x = self.dropout(x)
        x = self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        # print(layer.bias)
        # if layer.bias == None:
        #     print("YESSSSS")
        # elif hasattr(layer, "bias"):
        if not (layer.bias is None):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            for i, (batch, labels, filenames) in enumerate(self.train_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # Compute the forward pass of the model
                logits = self.model.forward(batch)

                # Compute the loss using self.criterion and store it
                loss = self.criterion(logits, labels)

                # Compute the backward pass
                loss.backward()

                # Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        # key = filename -> { label: x, preds = []}
        results = {}
        total_loss = 0

        # Turn on evaluation mode for network. This changes the behaviour of
        # dropout and batch normalisation during validation.
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (batch, labels, filenames) in enumerate(self.test_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()

                # Populate dictionary with scores for each segment in this batch assigned to the filename
                for j, filename in enumerate(filenames):
                    current_logits = logits[j].cpu().tolist()
                    if filename not in results:
                        results[filename] = {
                            "label" : labels[j],
                            "logits" : [current_logits],
                            "prediction"  : -1
                        }
                    else:
                        results[filename]["logits"].append(current_logits)
        
        # Take the average across each class score for each file to get a prediction
        results = compute_predictions(results)

        accuracy = compute_file_accuracy(results)
        per_class_accuracies = compute_file_per_class_accuracies(results)
        average_loss = total_loss / len(self.test_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        print(f"per class accuracies: {per_class_accuracies}")


def compute_predictions(results: dict) -> dict:

    for filename in results:
        # Convert list ot logits for this file to numpy array
        results[filename]["logits"] = np.asarray(results[filename]["logits"])
        
        # Average scores across segments for each class
        results[filename]["logits"] = np.mean(results[filename]["logits"], axis=0)
        
        # Get index of highest scoring class
        results[filename]["prediction"] = results[filename]["logits"].argmax(-1)

    return results


def compute_file_accuracy(results : dict) -> float:
    correct = 0
    for filename in results:
        if (results[filename]["label"] == results[filename]["prediction"]):
            correct += 1
    
    return correct/len(results.keys())

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_file_per_class_accuracies(results : dict) -> float:

    class_accuracies = {}

    for filename in results:
        label = results[filename]["label"].item()

        if label not in class_accuracies.keys():
            class_accuracies[label] = []
        
        if label == results[filename]["prediction"]:
            class_accuracies[label].append(1)
        else:
            class_accuracies[label].append(0)

    for label in class_accuracies:
        accuracy_count = class_accuracies[label]
        class_accuracies[label] = sum(accuracy_count)/len(accuracy_count)

    return class_accuracies

def compute_per_class_accuracies(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:

    class_accuracies = {}
    classes = np.unique(labels)

    # Go through each class available
    for c in classes:
        # Find all correct predictions
        matches = labels[labels == preds]
        # Extract all correct predicitons for given class
        class_matches = (matches == c).sum()
        # Calculate class accuracy
        class_accuracy = float(class_matches / (labels == c).sum())
        class_accuracies[c] = class_accuracy

    return class_accuracies

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_bn_"
      f"mode={args.mode}_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
