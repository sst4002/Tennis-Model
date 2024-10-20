import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load and clean the dataset (combine years as you have provided them)
file_paths = [
    'atp_matches_2020.csv',
    'atp_matches_2021.csv',
    'atp_matches_2022.csv',
    'atp_matches_2023.csv',
    'atp_matches_2024.csv'
]
dataframes = [pd.read_csv(file) for file in file_paths]
matches_df = pd.concat(dataframes, ignore_index=True)

# Fill missing values
numerical_columns = matches_df.select_dtypes(include=['float64', 'int64']).columns
matches_df[numerical_columns] = matches_df[numerical_columns].fillna(matches_df[numerical_columns].mean())
categorical_columns = matches_df.select_dtypes(include=['object']).columns
matches_df[categorical_columns] = matches_df[categorical_columns].fillna('Unknown')

# Drop unnecessary columns
# Drop unnecessary columns
drop_columns = ['tourney_id', 'tourney_name', 'winner_id', 'loser_id', 'score', 
                'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc']
matches_df_cleaned = matches_df.drop(columns=drop_columns)

# Fill missing values in numerical columns (e.g., 'winner_seed', 'loser_seed')
# First, we'll convert those columns to numeric, forcing errors (like strings) to NaN, then impute
matches_df_cleaned['winner_seed'] = pd.to_numeric(matches_df_cleaned['winner_seed'], errors='coerce')
matches_df_cleaned['loser_seed'] = pd.to_numeric(matches_df_cleaned['loser_seed'], errors='coerce')

# If there are other columns with missing values, you can fill them as well (for simplicity)
imputer = SimpleImputer(strategy='mean')
matches_df_cleaned[['winner_seed', 'loser_seed']] = imputer.fit_transform(matches_df_cleaned[['winner_seed', 'loser_seed']])

# Encode categorical variables with one-hot encoding
matches_df_cleaned = pd.get_dummies(matches_df_cleaned, columns=['surface', 'round', 'tourney_level', 
                                                                 'winner_hand', 'loser_hand', 
                                                                 'winner_entry', 'loser_entry'], drop_first=True)

# Target column: 1 if winner, 0 otherwise (you may need to adjust this logic depending on the actual target)
matches_df_cleaned['target'] = 1  # Make sure to modify this logic to reflect match outcomes if needed.

# Prepare the features (X) and target (y)
X = matches_df_cleaned.drop(columns=['target']).values
y = matches_df_cleaned['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure that X_train is numeric
print(X_train.dtype)  # Should print 'float64' or 'int64'

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Adding sequence length dimension
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# LSTM Model
class TennisLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TennisLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 64)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(0), 64)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Model parameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 1
output_size = 1

# Initialize the model, loss function, and optimizer
model = TennisLSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')
