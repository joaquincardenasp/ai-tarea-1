import torch
import torch.nn as nn
import torch.optim as optim

#Usar para BCELoss y Sigmoid:
def train_one_epoch(model, train_loader, optimizer, loss_function):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Salida es [batch_size, 1], usar squeeze() para que sea [batch_size]
        outputs = model(inputs).squeeze()
        # BCELoss espera salida [batch_size] y etiquetas float [batch_size]
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        # Calcular preds con umbral 0.5 para accuracy
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate(model, val_loader, loss_function):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Salida es [batch_size, 1], usar squeeze()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            # Calcular preds con umbral 0.5 para accuracy
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

"""
#Usar para CrossEntropyLoss y Softmax:
def train_one_epoch(model, train_loader, optimizer, loss_function):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Output shape is [batch_size, 2], no squeeze needed
        outputs = model(inputs)
        # CrossEntropyLoss takes raw scores (logits) and long labels
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        # Get predictions by finding the index of the max score
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate(model, val_loader, loss_function):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Output shape is [batch_size, 2], no squeeze needed
            outputs = model(inputs)
            # CrossEntropyLoss takes raw scores (logits) and long labels
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            # Get predictions by finding the index of the max score
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total"""