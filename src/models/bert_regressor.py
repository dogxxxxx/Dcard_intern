import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from bert_dataset_model import BertRegressor, MyDataset


config = {
    'train_path': 'data\\intern_homework_train_dataset.csv',
    'val_path': 'data\\intern_homework_public_test_dataset.csv',
    'seed': 42,
    'lr': 2e-5,
    'epochs': 5,
    'dropout': 0.3,
    'save_path': 'src\\models\\best_bert_regressor.pt'
}
train_data = pd.read_csv(config['train_path'])[['title', 'like_count_24h']]
val_data = pd.read_csv(config['val_path'])[['title', 'like_count_24h']]
train_data['labels'] = train_data['like_count_24h']
val_data['labels'] = val_data['like_count_24h']
train_data = train_data.drop('like_count_24h', axis=1)
val_data = val_data.drop('like_count_24h', axis=1)


train_set = MyDataset(df=train_data)
data_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_set = MyDataset(df=val_data)
val_loader = DataLoader(val_set, batch_size=16)
# Initialize the model and optimizer
model = BertRegressor()
optimizer = Adam(model.parameters(), lr=config['lr'])

# Define the loss function
criterion = nn.MSELoss()
best_accuracy=0

for epoch in range(config['epochs']):
    # Training loop
    running_loss = 0.0
    model.train()
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(data_loader)
    
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch+1}"):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            predicted = torch.round(outputs.squeeze())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}, val accuracy: {accuracy:.2f}%")

    # Save the model if the validation accuracy improves
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), config['save_path'])
        print("New best model saved!")
