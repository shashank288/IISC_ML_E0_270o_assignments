import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def preprocess_data(
       X: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Function to fill the missing values
    """
    raise NotImplementedError

def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load data from csv file and split into train, val and
    test set. Relabel the targets to M=1, B=0 .

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # split into train, val and test set
    train_size = int(0.8 * len(df)) 
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    raise NotImplementedError
