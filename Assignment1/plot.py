import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"
    # for i in [True, False]:
    #     for j in range(1, 6):
    #         assert os.path.exists(os.path.join(args.logs_path, f"run_{j}_{i}.npy")),\
    #             f"File run_{j}_{i}.npy not found in {args.logs_path}"
    # TODO: Load data and plot the standard means and standard deviations of
    # the accuracies for the two settings (active and random strategies)
    # TODO: also ensure that the files have the same length
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())
