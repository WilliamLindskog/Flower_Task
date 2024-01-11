from typing import Callable, Dict, OrderedDict
import torch
from torch.cuda import is_available
from torch.utils.data import DataLoader
from hydra.utils import call
from omegaconf import DictConfig

from flwr.client import NumPyClient
from flwr.common.typing import Scalar

from flwr_datasets import FederatedDataset

from task.utils import train, test
from task.dataset_preparation import apply_transforms

class FlowerClient(NumPyClient):
    """Flower client implementing FedAvg.
    
    Parameters
    ----------
    client_data : FederatedDataset
        The client data.
    device : torch.device
        The device to train on.
    cid : str
        The client id.
    cfg : DictConfig
        The configuration.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        client_data: FederatedDataset,
        device: torch.device,
        cid: str,
        cfg: DictConfig,
    ) -> None:
        # Initialize model
        self.net = call(cfg.model)
        self.net.to(device)

        self.client_data = client_data
        self.device = device
        self.cid = int(cid)
        self.cfg = cfg

        # Load data
        self.trainloader, self.testloader = self.load_data()

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        epochs = int(config["epochs"])
        lr = float(config["learning_rate"])

        self.set_parameters(parameters)
        # Get size of parameters in bytes 
        train(self.net, self.trainloader, self.device, epochs, lr)
        final_p_np = self.get_parameters({})

        return final_p_np, len(self.trainloader), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
    
    def load_data(self,):
        """Load partition CIFAR10 data."""
        # Divide data on each node: 80% train, 20% test
        partition_train_test = self.client_data.train_test_split(test_size=0.2)

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=self.cfg.batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=self.cfg.batch_size)
        return trainloader, testloader
    

def get_client_fn(fds: FederatedDataset, cfg: DictConfig,) -> Callable[[str], FlowerClient]:
    """ Return a function which creates a Flower client. 

    Parameters
    ----------
    fds : FederatedDataset
        The federated dataset to use.
    cfg : DictConfig
        The configuration.
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization.
        
        Parameters
        ----------
        cid : str
            The client id.

        Returns
        -------
        FlowerClient
            The Flower client representing a single organization.
        """
        # Load model
        client_data = fds.load_partition(int(cid), "train")
        client_data = client_data.with_transform(apply_transforms)
        device = torch.device("cuda:0" if is_available() else "cpu")
        return FlowerClient(client_data, device, cid, cfg,)

    return client_fn