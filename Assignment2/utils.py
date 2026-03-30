import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def preprocess_data(
       X: np.ndarray
) -> np.ndarray:
    """
    Function to fill the missing values
    """
    raise NotImplementedError

def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from csv file and split into train and val set.
    Relabel the targets to M=1, B=0.

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (X_train, y_train, X_val, y_val)
    """
    # load data
    df = pd.read_csv(path)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # TODO: Relabel the targets to M=1, B=0
    
    # TODO: split into train and val set (70/30 stratified)
    raise NotImplementedError