import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from utils import get_data

# ==========================================
# 1. DATA PREPARATION 
# ==========================================
def load_and_prep_data(filepath, seed):
    """Loads dataset, splits, standardizes, and converts to PyTorch Tensors."""
    X_train, y_train, X_val, y_val = get_data(filepath, seed)

    # Standardize features (Crucial for Neural Networks!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Reshape to column vector
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_train.shape[1]

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        # TODO: Define a single linear layer (Linear mapping from input_dim to 1 output)
        pass 

    def forward(self, x):
        # TODO: Pass x through the linear layer. 
        # Hint: We will use BCEWithLogitsLoss later, so you DO NOT need to apply a Sigmoid here. Just return the raw logits.
        pass

class MLPTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # TODO: Define the layers for a Multi-Layer Perceptron.
        # e.g., Linear(input->hidden) -> ReLU -> Linear(hidden->1)
        pass

    def forward(self, x):
        # TODO: Implement the forward pass through your hidden layers and activation functions.
        pass

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=1000):
    # TODO: Define the loss function. Use nn.BCEWithLogitsLoss() for numerical stability.
    criterion = None 
    
    # TODO: Define the optimizer. Use optim.SGD or optim.Adam. Pass in model.parameters() and the learning_rate.
    optimizer = None 

    train_losses = []

    for epoch in range(epochs):
        model.train() # Set model to training mode
        
        # --- 1. Forward Pass ---
        # TODO: Pass X_train through the model to get predictions
        outputs = None 
        
        # --- 2. Compute Loss ---
        # TODO: Calculate the loss between outputs and y_train
        loss = None 
        
        # --- 3. Backward Pass & Optimization ---
        # TODO: Zero the gradients (optimizer.zero_grad())
        # TODO: Backpropagate the error (loss.backward())
        # TODO: Update the weights (optimizer.step())
        
        train_losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return train_losses

# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Make sure data.csv is in the same directory)
    # X_train, y_train, X_val, y_val, input_dim = load_and_prep_data('data.csv', seed)
    
    # 2. Instantiate Model, select one at a time
    # model_lr = LogisticRegressionTorch(input_dim)
    # model_mlp = MLPTorch(input_dim)
    
    # 3. Train Model
    # losses = train_model(model_lr/mlp, X_train, y_train, X_val, y_val, learning_rate=0.1, epochs=500)
    
    # 4. Evaluation (Add code here to calculate test F1, Precision, Recall)
