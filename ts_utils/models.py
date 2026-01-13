"""
This file contains the model classes necessary for a convolutional orthogonal autoencoder (COAE).
The encoder and decoder are set up to define the entire COAE with an array of hyperparameters and 
architectures. The loss function for the COAE is a combination of mean square error (MSE) and 
ortogonal loss.
"""
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_ts import denormalize
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional, Union, Any


class lstm_layer(nn.Module):
    """Single LSTM layer"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_size: int = 1,
        device: torch.device = None,
        batch_first: bool = True,
        return_all: bool = True,
        stateful: bool = True,
        zero_initial_state: bool = True,
    ):
        """initialize lstm_layer module

        :param input_size: size of the input vector
        :type input_size: int
        :param hidden_size: size of the output vector
        :type hidden_size: int
        :param batch_size: batch size for the model, defaults to 1
        :type batch_size: int, optional
        :param device: device to use for the LSTM state (should be same as overall model), defaults to None
        :type device: torch.device, optional
        :param batch_first: is the batch dimension first, defaults to True
        :type batch_first: bool, optional
        :param return_all: do you want to return the look-back time dimension, defaults to True
        :type return_all: bool, optional
        :param stateful: is the model stateful, defaults to True
        :type stateful: bool, optional
        :param zero_initial_state: do you want to use zeros for the initial cell states (else random), defaults to True
        :type zero_initial_state: bool, optional
        """
        super(lstm_layer, self).__init__()
        self._layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._batch_first = batch_first
        self._return_all = return_all
        self._stateful = stateful
        self._zero_initial_state = zero_initial_state
        self._shape = (batch_size, 1, hidden_size)
        self._device = device
        self.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through the LSTM layer

        :param x: input tensor
        :type x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        x, (hn, cn) = self._layer(x, (self._hn, self._cn))
        if self._stateful:
            self._hn, self._cn = hn.detach(), cn.detach()
        if not self._return_all:
            x = x[:, -1, :]
        return x

    def reset_state(self) -> None:
        """reset the state of the LSTM layer"""
        self._hn = torch.zeros(self._shape) if self._zero_initial_state else torch.rand(self._shape)
        self._cn = torch.zeros(self._shape) if self._zero_initial_state else torch.rand(self._shape)
        if self._device is not None:
            self._hn = self._hn.to(self._device)
            self._cn = self._cn.to(self._device)


class LSTM(nn.Module):
    """Encoder section of the COAE"""

    def __init__(self, config: Dict[str, Any], device, stateful: bool = True):
        """initializes Encoder object

        :param config:
        :type config: Dict[str, Any]
        :param device: device to use for the LSTM state (should be same as overall model)
        :type device: torch.device
        :param stateful: is the model stateful, defaults to True
        :type: stateful: bool, optional
        """
        super(LSTM, self).__init__()
        assert device is not None
        batch_size = config.get("batch_size", 1)
        num_lstm_nodes = [
            config.get("num_inputs_total", 16),
            config.get("LSTM_nodes_0", 32),
            config.get("LSTM_nodes_1", 32),
        ][: config.get("num_lstm_layers", 1) + 1]
        # num_lstm_nodes += [config.get("Dense_nodes_0", 64)]
        lstm_activations = [config.get("LSTM_activation_0", "tanh"), config.get("LSTM_activation_1", "tanh")][
            : config.get("num_lstm_layers", 1)
        ]
        num_dense_nodes = [num_lstm_nodes[-1]]
        num_dense_nodes += [
            config.get("Dense_nodes_0", 32),
            config.get("Dense_nodes_1", 32),
            config.get("Dense_nodes_1", 32),
        ][: config.get("num_dense_layers", 1)]
        num_dense_nodes += [config.get("bottleneck_size", 10)]
        dense_activations = [
            config.get("Dense_activation_0", "tanh"),
            config.get("Dense_activation_1", "tanh"),
            config.get("Dense_activation_2", "tanh"),
        ][: config.get("num_dense_layers", 1)]
        dropout = [config.get("dropout_0", 0.0), config.get("dropout_1", 0.0), config.get("dropout_2", 0.0)][
            : config.get("num_dense_layers", 1)
        ]
        self._setup_activation_dict()
        self._build_layers(
            batch_size,
            stateful,
            config.get("num_lstm_layers", 1),
            num_lstm_nodes,
            lstm_activations,
            config.get("num_dense_layers", 1),
            num_dense_nodes,
            dense_activations,
            dropout,
            config.get("zero_inital_state", True),
            device,
        )

    def _setup_activation_dict(self) -> None:
        """creates a dictionary that converts activation function names to the functional object"""
        self._activation_dict = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
            "softsign": nn.Softsign(),
        }

    def _build_layers(
        self,
        batch_size: int,
        stateful: bool,
        num_lstm_layers: int,
        num_lstm_nodes: List[int],
        lstm_activations: List[int],
        num_dense_layers: int,
        num_dense_nodes: List[int],
        dense_activations: List[int],
        dropout: List[int],
        zero_initial_state: bool,
        device: torch.device,
    ) -> None:
        """builds all layers depending on the configuration

        :param batch_size: batch size for training/evaluation
        :type batch_size: int
        :param stateful: is the model stateful
        :type stateful: bool
        :param num_lstm_layers:
        :type num_lstm_layers: int
        :param num_lstm_nodes:
        :type num_lstm_nodes: List[int]
        :param lstm_activations:
        :type lstm_activations: List[int]
        :param num_dense_layers:
        :type num_dense_layers: int
        :param num_dense_nodes:
        :type num_dense_nodes: List[int]
        :param dense_activations:
        :type dense_activations: List[int]
        :param dropout:
        :type dropout: List[str]
        :param zero_initial_state: do you want to use zeros for the initial LSTM states
        :type zero_initial_state: bool
        :param device: device to use for the LSTM state (should be same as overall model)
        :type device: torch.device
        """
        self._lstm_indices = []
        self._layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            self._layers.append(
                lstm_layer(
                    input_size=num_lstm_nodes[i],
                    hidden_size=num_lstm_nodes[i + 1],
                    batch_size=batch_size,
                    device=device,
                    batch_first=True,
                    return_all=False if (i + 1) == num_lstm_layers else True,
                    stateful=stateful,
                    zero_initial_state=zero_initial_state,
                )
            )
            self._lstm_indices.append(2 * i)
            self._layers.append(self._activation_dict[lstm_activations[i].lower()])
        for i in range(num_dense_layers):
            self._layers.append(
                nn.Linear(
                    in_features=num_dense_nodes[i],
                    out_features=num_dense_nodes[i + 1],
                )
            )
            self._layers.append(self._activation_dict[dense_activations[i].lower()])
            self._layers.append(nn.Dropout(dropout[i]))
        self._layers.append(
            nn.Linear(
                in_features=num_dense_nodes[-2],
                out_features=num_dense_nodes[-1],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """passes input through the model

        :param x: input tensor
        :type x: torch.Tensor
        :return: latent output mean and log variance
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        for layer in self._layers:
            x = layer(x)
        return x

    def loss(self, y: torch.Tensor, yhat: torch.Tensor) -> Tuple[torch.FloatTensor]:
        """loss function for the LSTM

        :param x: ground truth
        :type x: torch.Tensor
        :param y: model prediction
        :type y: torch.Tensor
        :return: loss for the batch
        :rtype: Tuple[torch.FloatTensor]
        """
        # compute basic MSE loss
        # print(f"y: {y.shape}")
        # print(f"yhat: {yhat.shape}")
        return torch.mean((y - yhat).pow(2))

    def reset_states(self) -> None:
        """resets the state of each LSTM layer"""
        for i in self._lstm_indices:
            self._layers[i].reset_state()