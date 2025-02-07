import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.naive_bayes import MultinomialNB

#Load and preprocess dataset
import pandas as pd

#Load dataset
emails = pd.read_csv('emails.csv')
#emails = emails.sample(frac=0.01, random_state=42)

#Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(emails['body']).toarray()
y = emails['label'].values

#Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Convert train and test sets to tensors so they can be used with MLP
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#Create DataLoader objects
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Implement MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#Create function to train MLP model
def train_model(model, loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

#Create function to test MLP model
def test_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, preds = torch.max(output, 1)
            y_true.extend(target.numpy())
            y_pred.extend(preds.numpy())
    return y_true, y_pred


#Define Hyperparameters
input_size = 1000
hidden_size = 128
num_classes = 2
learning_rate = 0.001
num_epochs = 10

mlp_model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

#Train MLP model
print("Training MLP...")
train_model(mlp_model, train_loader, criterion, optimizer, num_epochs)

#Test MLP model
print("Testing MLP...")
y_true_mlp, y_pred_mlp = test_model(mlp_model, test_loader)

#Evaluate MLP model
mlp_conf_matrix = confusion_matrix(y_true_mlp, y_pred_mlp)
mlp_accuracy = accuracy_score(y_true_mlp, y_pred_mlp)
mlp_precision = precision_score(y_true_mlp, y_pred_mlp)

print("MLP Confusion Matrix:\n", mlp_conf_matrix)
print(f"MLP Accuracy: {mlp_accuracy}")
print(f"MLP Precision: {mlp_precision}")


#Create and train Naive Bayes Model
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

#Test Naive Bayes Model
y_pred_nb = nb_model.predict(X_test)

#Evaluate Naive Bayes
nb_conf_matrix = confusion_matrix(y_test, y_pred_nb)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)

print("Naive Bayes Confusion Matrix:\n", nb_conf_matrix)
print(f"Naive Bayes Accuracy: {nb_accuracy}")
print(f"Naive Bayes Precision: {nb_precision}")

#Compare performance
print("\nComparison of Models:")
print(f"MLP Accuracy: {mlp_accuracy} vs Naive Bayes Accuracy: {nb_accuracy}")
print(f"MLP Precision: {mlp_precision} vs Naive Bayes Precision: {nb_precision}")
