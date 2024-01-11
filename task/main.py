from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from time import time
from torch.utils.data import DataLoader

from task.utils import plot_metric_from_history
from task.dataset import load_data
from task.dataset_preparation import apply_transforms

from flwr.server import ServerConfig
from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.simulation import start_simulation

@hydra.main(config_path="conf", config_name="task", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    # Print parsed config
    print(OmegaConf.to_yaml(cfg))
    assert cfg.num_clients > 0, "Number of clients must be greater than 0."
    assert cfg.clients_per_round <= cfg.num_clients, (
        "Number of clients per round must be less than or equal to the number of clients."
    )

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    fds = load_data(cfg)
    testdata = fds.load_full("test")
    testdata = testdata.with_transform(apply_transforms)
    testloader = DataLoader(testdata, batch_size=cfg.batch_size, shuffle=False)

    # Define clients and get evaluate function
    client_fn = call(cfg.client_fn, fds, cfg,)
    evaluate_fn = call(cfg.evaluate_fn, testloader, device, cfg)

    # Get strategy and server
    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)
    server = Server(strategy=strategy, client_manager=SimpleClientManager()) 

    # Start Simulation
    # measure time
    start_time = time()
    history = start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )
    end_time = time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed: {time_elapsed:.2f} seconds")

    # Plot results
    file_suffix: str = (
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_lr={cfg.learning_rate}"
    )
    save_path = HydraConfig.get().runtime.output_dir
    plot_metric_from_history(history, save_path, (file_suffix),)

if __name__ == "__main__":
    main()
