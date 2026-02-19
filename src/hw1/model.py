from numpy.typing import ArrayLike
from typing import Any
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


class BinaryClfModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=1,
        n_layers=3,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_lin_layers = n_layers
        self.activation = activation

        layers = [
            nn.Linear(in_features=input_size, out_features=hidden_size),
            activation(),
        ]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(activation())
        layers.append(nn.Linear(in_features=hidden_size, out_features=output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        logits = self.layers(X)
        return logits
