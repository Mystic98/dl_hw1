from hw1.evaluate import evaluate
from hw1.utils import find_device
from IPython.display import clear_output
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fontTools.misc.plistlib import Data
from tqdm import tqdm
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
from hw1.model import BinaryClfModel


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: BinaryClfModel,
    optim=optim.Adam,
    learning_rate=0.01,
    n_epochs=20,
    eval_step=5,
):
    device = find_device()
    optimizer = optim(params=model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()

    loss_history = []
    f1_history = []
    eval_loss_history = []
    eval_f1_history = []

    for epoch in tqdm(range(n_epochs)):
        size = 0
        batch_loss = 0
        batch_f1score = 0
        for X_batched, y_batched in train_dataloader:
            batch_size = X_batched.shape[0]
            size += batch_size

            optimizer.zero_grad()
            logits = model(X_batched.to(device))
            loss = criterion(logits, y_batched.to(device))
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() * batch_size

            y_pred = (torch.sigmoid(logits) >= 0.5).int().cpu()
            batch_f1score += (
                f1_score(y_batched.cpu().numpy(), y_pred.numpy()) * batch_size
            )

        f1_epoch = batch_f1score / size
        loss_epoch = batch_loss / size
        f1_history.append(f1_epoch)
        loss_history.append(loss_epoch)

        val_loss, val_score = evaluate(val_dataloader, model)
        model.train()

        eval_loss_history.append(val_loss)
        eval_f1_history.append(val_score)
        if epoch % eval_step == 0:
            clear_output()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # ----------------
            # Plot 1: Loss
            # ----------------
            epochs = np.arange(epoch + 1)
            axes[0].plot(epochs, loss_history, label="train loss", color="blue")
            axes[0].plot(epochs, eval_loss_history, label="val loss", color="orange")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("loss")
            axes[0].legend()

            # ----------------
            # Plot 2: F1
            # ----------------
            axes[1].plot(epochs, f1_history, label="train f1 (epoch)", color="blue")
            axes[1].plot(
                epochs, eval_f1_history, label="val f1 (epoch)", color="orange"
            )
            axes[1].set_title("F1 score")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("F1")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
