import os
import argparse
import numpy as np
from utils import *
from model import *

def main(args: argparse.Namespace):
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no+args.run_id)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    
    X_train = preprocess_data(X_train)
    # TODO: Preprocess validation split
    print("Data Loaded and Preprocessed")

    # Train the model
    model = GaussianNaiveBayes(var_smoothing=args.smoothing)
    accs = []
    total_items = 20
    batch_size = 10
    
    idxs = np.arange(total_items)
    remaining_idxs = np.setdiff1d(np.arange(X_train.shape[0]), idxs)
    
    # Train the model
    for i in range(1, 40):
        if len(idxs) > X_train.shape[0]:
            break
            
        X_train_batch = X_train[idxs]
        y_train_batch = y_train[idxs]
        
        if i == 1:
            model.fit(X_train_batch, y_train_batch)
        else:
            model.fit(X_train_batch, y_train_batch, update=True)
            
        y_preds = model.predict(X_val)
        val_acc = np.mean(y_preds == y_val)
        print(f"{total_items} items - Val acc: {val_acc}")
        accs.append(val_acc)

        if len(remaining_idxs) == 0:
            break
            
        current_batch_size = min(batch_size, len(remaining_idxs))

        if args.is_active:
            raise NotImplementedError
        else:
            idxs = np.concatenate([idxs, remaining_idxs[:current_batch_size]])
            remaining_idxs = remaining_idxs[current_batch_size:]
            
        total_items += current_batch_size
        
    accs = np.array(accs)
    os.makedirs(args.logs_path, exist_ok=True)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train1.csv")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--smoothing", type=float, default=1e-9)
    main(parser.parse_args())