import torch 
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD

from typing import Tuple, Dict, Callable, Optional
from pathlib import Path
from flwr.server.history import History

import matplotlib.pyplot as plt
import numpy as np

# Define fit_config
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations.
    
    Returns
    -------
    Callable[[int], Dict[str, str]]
        A function which returns training configurations.
    """

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs.
        
        Returns
        -------
        Dict[str, str]
            A configuration with static batch size and (local) epochs.
        """
        config = {
            "learning_rate": str(0.01),
            "batch_size": str(64),
            "lr_decay" : str(0.99),
            "epochs" : str(2)
        }
        return config

    return fit_config

def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device: torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    lr : float
        The learning rate to use for training.

    Returns
    -------
    None
    """
    # Get training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=lr,)
    # Train model
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(net, trainloader, device, criterion, optimizer,)
    
def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
) -> nn.Module:
    """Train the network on the training set for one epoch.
    
    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device: torch.device
        The device on which to train the network.
    criterion : nn.Module
        The loss function to use for training.
    optimizer : Optimizer
        The optimizer to use for training.

    Returns
    -------
    nn.Module
        The trained neural network.
    """
    for batch in trainloader:
        data, target = batch["image"].to(device), batch["label"].to(device)
        target = target.long()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return net

def test(
    net: nn.Module, 
    testloader: DataLoader, 
    device: torch.device, 
) -> Tuple[float, float]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:       
            images, labels = batch["image"].to(device), batch["label"].to(device)
            # images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_dict = hist.metrics_centralized
    print(metric_dict)

    _, accuracy = zip(*metric_dict["accuracy"])
    accuracy = tuple(x for x in accuracy)
    rounds_loss, values_loss = zip(*hist.losses_centralized)
    # make tuple of normal floats instead of tensors
    values_loss = tuple(x for x in values_loss)

    # Add grid to plot
    plt.style.use("ggplot")

    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(accuracy))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    plt.xlabel("Rounds")

    plt.savefig(Path(save_plot_path) / Path(f"mnist_metrics{suffix}.png"))
    plt.close()