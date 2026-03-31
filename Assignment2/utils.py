import pickle
import random
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple, Optional


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess_data(
       X: np.ndarray,
       f_medians: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill missing values via median imputation. Computes medians from X if f_medians is None."""
    if f_medians is None:
        f_medians = np.nanmedian(X, axis=0)
    l_X_filled = X.copy()
    l_nan_indices = np.where(np.isnan(l_X_filled))
    l_X_filled[l_nan_indices] = np.take(f_medians, l_nan_indices[1])
    return l_X_filled, f_medians

def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load csv, relabel M=1/B=0, stratified 70/30 split."""
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    l_y = (df['diagnosis'] == 'M').astype(int).values
    l_X = df.drop('diagnosis', axis=1).values.astype(float)

    l_idx_class0 = np.where(l_y == 0)[0]
    l_idx_class1 = np.where(l_y == 1)[0]

    l_n_train_0 = int(0.7 * len(l_idx_class0))
    l_n_train_1 = int(0.7 * len(l_idx_class1))

    l_train_idx = np.concatenate([l_idx_class0[:l_n_train_0], l_idx_class1[:l_n_train_1]])
    l_val_idx = np.concatenate([l_idx_class0[l_n_train_0:], l_idx_class1[l_n_train_1:]])

    return l_X[l_train_idx], l_y[l_train_idx], l_X[l_val_idx], l_y[l_val_idx]