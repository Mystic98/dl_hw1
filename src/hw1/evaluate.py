from hw1.model import BinaryClfModel
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
from hw1.utils import find_device


@torch.no_grad()
def evaluate(eval_dataloader: DataLoader, model: BinaryClfModel):
    device = find_device()
    y_true = []
    y_pred = []
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.eval()
    size = 0
    batches_loss = 0
    for X_batched, y_batched in eval_dataloader:
        batch_size = X_batched.shape[0]
        logits = model(X_batched.to(device))
        batches_loss += criterion(logits, y_batched.to(device)).item() * batch_size
        size += batch_size

        y_pred += (torch.sigmoid(logits) >= 0.5).int().cpu().tolist()
        y_true += y_batched.cpu().tolist()
    loss = batches_loss / size
    score = f1_score(y_true, y_pred)
    return loss, score
