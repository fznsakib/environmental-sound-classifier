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
    default=20,
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
    default=8,
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
    # Data augmentation goes here

    mode = args.mode

    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset("data/UrbanSound8K_train.pkl", mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset("data/UrbanSound8K_test.pkl", mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.sgd_momentum)

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
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(dropout)

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.initialise_layer(self.conv1)
        self.batchNorm1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.initialise_layer(self.conv2)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv3)
        self.batchNorm3 = nn.BatchNorm2d(64)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.initialise_layer(self.conv4)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ## First fully connected layer
        self.fc1 = nn.Linear(13440, 1024) # 13440 = 10 * 21 * 64
        self.initialise_layer(self.fc1)
        # self.batchNorm5 = nn.BatchNorm1d(1024)

        ## Second fully connected layer
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # First convolutional layer pass
        x = self.conv1(images)
        x = self.batchNorm1(x)
        x = F.relu(x)

        # Second convolutional layer pass
        x = self.conv2(self.dropout(x))
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third convolutional layer pass
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        # Fourth convolutional layer pass
        x = self.conv4(self.dropout(x))
        x = self.batchNorm4(x)
        x = F.relu(x)
        x = self.pool4(x)

        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ## (batch_size, 4096)
        x = torch.flatten(x, start_dim=1)

        # First fully connected layer pass
        x = self.fc1(self.dropout(x))
        # x = self.batchNorm5(x)
        x = F.sigmoid(x)

        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fc2(self.dropout(x))
        x = F.softmax(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
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
            # print(self.train_loader)
            # print(self.train_loader[0])

            for i, (input, labels, filename) in enumerate(self.train_loader):
                input = input.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                ## TASK 1: Compute the forward pass of the model, print the output shape
                ## and quit the program
                ## TASK 7: Rename `output` to `logits`
                logits = self.model.forward(input)

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass
                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
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
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.test_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(np.array(results["labels"]), np.array(results["preds"]))
        average_loss = total_loss / len(self.test_loader)

        per_class_accuracies = compute_per_class_accuracies(np.array(results["labels"]), np.array(results["preds"]))

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
      f"dropout={args.dropout}_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"momentum={args.sgd_momentum}_" +
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
