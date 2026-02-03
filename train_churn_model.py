import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Custom Dataset
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # 3. Load Data
    try:
        df = pd.read_csv('shopping_mall_churn.csv')
    except FileNotFoundError:
        print("Error: 'shopping_mall_churn.csv' not found. Please run 'generate_churn_data.py' first.")
        return

    print("Data Columns:", df.columns.tolist())
    
    # Features and Target
    # Input: Purchase_Count_3M, Days_Since_Last_Login, Avg_Transaction_Amt, Age, Gender, Claim_Count
    feature_cols = ['Purchase_Count_3M', 'Days_Since_Last_Login', 'Avg_Transaction_Amt', 'Age', 'Gender', 'Claim_Count']
    target_col = 'Churn'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 4. Preprocessing
    # Scale numerical features. Gender is 0/1, fits fine with scaler or without. Scaling everything is safe.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Handle Data Imbalance (Using WeightedRandomSampler)
    # Count classes in training set
    class_counts = np.bincount(y_train)
    class_0_count = class_counts[0]
    class_1_count = class_counts[1]
    
    print(f"\nTraining Data Balance: Retain(0)={class_0_count}, Churn(1)={class_1_count}")
    
    # Calculate weights for each class (inverse of frequency)
    weight_for_0 = 1.0 / class_0_count
    weight_for_1 = 1.0 / class_1_count
    
    # Assign weight to each sample in the dataset
    sample_weights = np.array([weight_for_1 if t == 1 else weight_for_0 for t in y_train])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    # Create Sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create Datasets
    train_dataset = ChurnDataset(X_train, y_train)
    test_dataset = ChurnDataset(X_test, y_test)
    
    # Create DataLoaders
    # Note: When using sampler, shuffle must be False
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 6. Define Model
    class ChurnModel(nn.Module):
        def __init__(self, input_dim):
            super(ChurnModel, self).__init__()
            self.layer1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(32, 16)
            self.relu3 = nn.ReLU()
            self.output = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid() # As requested: Sigmoid at the end
            
        def forward(self, x):
            x = self.relu1(self.layer1(x))
            x = self.relu2(self.layer2(x))
            x = self.relu3(self.layer3(x))
            x = self.sigmoid(self.output(x))
            return x

    input_dim = X_train.shape[1]
    model = ChurnModel(input_dim).to(device)
    
    # 7. Loss and Optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy for probability output (0~1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 8. Training Loop
    epochs = 50
    print("\nStarting Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy calc
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Acc: {correct/total:.2%}")
            
    # 9. Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For metrics
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    print(f"\nFinal Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Final Test Accuracy: {correct/total:.2%}")
    
    # Simple confusion matrix print
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Retain(0)', 'Churn(1)']))
    
    # 10. Save
    torch.save(model.state_dict(), 'churn_model.pth')
    joblib.dump(scaler, 'churn_scaler.pkl')
    print("Model saved to churn_model.pth")

if __name__ == "__main__":
    main()
