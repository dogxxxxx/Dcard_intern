import torch
from torch.optim.adam import Adam
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
import numpy as np

def train_gru(model, train_loader, val_loader, lr, epochs, patience=10):
    criterion = MeanAbsolutePercentageError()
    optimizer = Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    train_loss_history=[]
    val_loss_history=[]
    for epoch in range(epochs):
        model.train()
        train_loss=0
        for inputs, label in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
        train_loss_history.append(train_loss/len(train_loader))
        
        model.eval()
        val_loss=0
        with torch.no_grad():
            for inputs, label in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_loss+=loss.item()
            val_loss_history.append(val_loss/len(val_loader))

        # early stopping    
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            counter=0
        else:
            counter+=1
            if counter>=patience:
                print('Early stopping at epoch: ', epoch+1)
                break
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')
    plt.plot(train_loss_history, 'b')
    plt.plot(val_loss_history, 'r')
    plt.show()
    return best_model

def test_gru(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            preds = model(inputs).numpy()
    return preds

def train_mlp(model, train_loader, val_loader, epochs, lr, patience=20):
    criterion = MeanAbsolutePercentageError()
    optimizer = Adam(model.parameters(), lr=lr)
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for data, label in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        train_loss_history.append(train_loss/len(train_loader))
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, label in val_loader:
                outputs = model(data)
                loss = criterion(np.round(outputs), label)
                val_loss+=loss.item()
            val_loss_history.append(val_loss/len(val_loader))
        
        # early stopping
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            counter=0
        else:
            counter+=1
            if counter>=patience:
                print('Early stopping at epoch: ', epoch+1)
                break
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')
    plt.plot(train_loss_history, 'b')
    plt.plot(val_loss_history, 'r')
    plt.legend(['train loss', 'validation loss'])
    plt.show()
    return best_model
        
def test_mlp(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            preds = model(inputs).numpy()
    return preds
