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

import os
from tqdm import tqdm


def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    load data from data folder and return a tuple X, y
    """
    data = pd.read_csv(path)
    X, y = data.drop(columns=["Class"]), data["Class"]
    return X, y


def split_data(X, y) -> Any:
    """
    split data into X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_new = np.asarray(X)
    y_new = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y_new, train_size=0.8, random_state=42, stratify=y_new
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.75, random_state=42, stratify=y_train
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_scaler(X_train: ArrayLike) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale(X: ArrayLike, scaler: StandardScaler) -> ArrayLike:
    """
    scale data using StandardScaler()
    """
    return scaler.transform(X)


class CustomDataset(Dataset):
    def __init__(self, X: ArrayLike, y: ArrayLike):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return self.X.shape[0]


def create_dataloader(
    X: ArrayLike, y: ArrayLike, batch_size=32, shuffle=True
) -> DataLoader:
    """
    create a dataset class from X and y and wrap it into DataLoader to be able to sample data
    """
    return DataLoader(
        dataset=CustomDataset(X, y), batch_size=batch_size, shuffle=shuffle
    )
