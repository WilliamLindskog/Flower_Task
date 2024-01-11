from omegaconf import DictConfig

from flwr_datasets import FederatedDataset

def load_data(cfg: DictConfig) -> FederatedDataset:
    """Return the dataloaders for the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config for the dataset.

    Returns
    -------
    Federated Dataset.
    """

    fds = FederatedDataset(dataset='mnist', partitioners={"train" : cfg.num_clients})
    return fds