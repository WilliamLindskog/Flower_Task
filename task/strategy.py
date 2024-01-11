from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns, Scalar, NDArrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Dict, Callable, Optional

class DiffLRStrategy(FedAvg):
    """Custom FedAvg which discards updates from stragglers.
    
    This strategy is based on the FedAvg strategy from Flower.
    It sets a permanent learning rate for half of the clients
    and a learning rate with decay for the other half.

    Parameters
    ----------
    min_fit_clients : int, optional
        The minimum number of clients to fit the model. Defaults to 2.
    min_available_clients : int, optional
        The minimum number of clients available for training. Defaults to 2.
    evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ], optional
        The evaluation function for the server. Defaults to None.
    on_fit_config_fn : Optional[Callable[[int], Dict[str, Scalar]]], optional
        The function which returns the configuration for the clients. Defaults to None.
    """
    def __init__(
        self,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    ) -> None:
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
        )

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all
                currently connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            
            # Set permanent learning rate for clients
            # or with learning rate decay
            config_permanent = config.copy()
            config_decay = config.copy()
            config_decay["learning_rate"] = str(
                # learning rate decay based on server round
                float(config_permanent["learning_rate"]) * float(config_permanent["lr_decay"]) ** server_round
            )	

        # Create FitIns for all clients
        fit_ins_permanent = FitIns(parameters, config_permanent)
        fit_ins_decay = FitIns(parameters, config_decay)

        # Sample all clients
        clients = client_manager.all()
        client_keys = list(clients.keys())
        
        # Split client keys into two groups
        len_clients = len(client_keys)
        client_keys_permanent = client_keys[:int(len_clients/2)]
        client_keys_decay = client_keys[int(len_clients/2):]

        # Return client/config pairs
        return [
            (clients[client_key], fit_ins_permanent) for client_key in client_keys_permanent
        ] + [
            (clients[client_key], fit_ins_decay) for client_key in client_keys_decay
        ]
