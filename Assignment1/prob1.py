import os

from utils import *
from model import *


def main(args: argparse.Namespace):
    # set seed for reproducibility
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
    model = GaussianNaiveBayes(alpha=args.smoothing)
    model.fit(X_train_vec, y_train)
    print("Model Trained")

    # Evaluate the trained model
    y_pred = model.predict(X_train_vec)
    print(f"Train Accuracy: {np.mean(y_pred == y_train)}")
    y_pred = model.predict(X_val_vec)
    print(f"Validation Accuracy: {np.mean(y_pred == y_val)}")
    # TODO: Note down the validation accuracy
    # TODO: Write and call functions for other evaluation metrics 

    # Load the test data
    if os.path.exists(f"{args.data_path}/X_test{args.intermediate}"):
        X_test_vec = pickle.load(open(
            f"{args.data_path}/X_test{args.intermediate}", "rb"))
        print("Preprocessed Test Data Loaded")
    else:
        X_test = pd.read_csv(
            f"{args.data_path}/X_test_{args.sr_no}.csv", header=None
        ).values.squeeze()
        print("Test Data Loaded")
    preds = model.predict(X_test)
    with open(f"predictions.csv", "w") as f:
        for pred in preds:
            f.write(f"{pred}\n")
    print("Predictions Saved to predictions.csv")
    print("You may upload the file at http://10.192.30.174:8000/submit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train.csv")
    main(parser.parse_args())
