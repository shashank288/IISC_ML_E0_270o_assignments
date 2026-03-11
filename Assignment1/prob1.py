import os
import argparse
import pickle
import numpy as np
import pandas as pd
from utils import *
from model import *

def main(args: argparse.Namespace):
    set_seed(args.sr_no)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")

    # Preprocess the data
    X_train = preprocess_data(X_train)
    #TODO: Write preprocessing for validation split
    print("Data Preprocessed")


    # Train the model
    model = GaussianNaiveBayes(var_smoothing=args.smoothing)
    model.fit(X_train, y_train)
    print("Model Trained")

    # Evaluate the trained model
    y_pred = model.predict(X_train)
    print(f"Train Accuracy: {np.mean(y_pred == y_train)}")
    y_pred = model.predict(X_val)
    print(f"Validation Accuracy: {np.mean(y_pred == y_val)}")
    
    # TODO: Note down the validation accuracy
    # TODO: Write and call functions for precision, recall, f1, roc-auc
    # TODO: Plot ROC and Precision-Recall curves
    # TODO: Provide confusion matrix and find threshold for >0.95 recall

    # Load the test data
    if os.path.exists(f"{args.data_path}/test1.csv"):
        # TODO: Load test1.csv, preprocess, predict, and report final metrics
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train1.csv")
    parser.add_argument("--smoothing", type=float, default=1e-9)
    main(parser.parse_args())