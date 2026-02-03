import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# 1. Custom Dataset Class
class CarPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) # Reshape to (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Loading & Preprocessing
    df = pd.read_csv('used_car_prices.csv')
    
    # Features and Target
    X = df.drop('Price', axis=1)
    y = df['Price'].values

    # Preprocessing Pipeline
    # Categorical features need One-Hot Encoding
    # Numerical features need Scaling
    categorical_features = ['Brand', 'Model', 'Fuel_Type', 'Accident_History']
    numerical_features = ['Year', 'Mileage', 'Engine_Size']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Scale target variable (Price) for better convergence usually, 
    # but for simplicity let's keep it raw or scale it manually if loss explodes.
    # Let's scale price by dividing by 1,000,000 (Million Won) for numerical stability
    y_scaled = y / 1000000.0

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)

    # Create Datasets and DataLoaders
    train_dataset = CarPriceDataset(X_train, y_train)
    test_dataset = CarPriceDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Define Neural Network
    class CarPriceModel(nn.Module):
        def __init__(self, input_dim):
            super(CarPriceModel, self).__init__()
            self.layer1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(32, 1) # Output: Predicted Price

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.layer2(x)
            x = self.relu2(x)
            x = self.layer3(x)
            return x

    input_dim = X_train.shape[1]
    model = CarPriceModel(input_dim).to(device)

    # 4. Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 100
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 6. Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"\nFinal Test MSE Loss: {test_loss/len(test_loader):.4f}")

    # Prediction Example
    print("\n--- Sample Predictions ---")
    with torch.no_grad():
        sample_inputs = torch.tensor(X_test[:5], dtype=torch.float32).to(device)
        sample_targets = y_test[:5]
        predictions = model(sample_inputs).cpu().numpy().flatten()
        
        for i in range(5):
            true_price = sample_targets[i] * 1000000
            pred_price = predictions[i] * 1000000
            print(f"True: {true_price:,.0f} KRW | Pred: {pred_price:,.0f} KRW | Diff: {abs(pred_price - true_price):,.0f}")


    # 7. Save Model and Preprocessor
    torch.save(model.state_dict(), 'used_car_price_model.pth')
    joblib.dump(preprocessor, 'used_car_price_preprocessor.pkl')
    print("\nModel and Preprocessor saved successfully.")
    print(" - Model: used_car_price_model.pth")
    print(" - Preprocessor: used_car_price_preprocessor.pkl")

if __name__ == "__main__":
    main()
