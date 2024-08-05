# completed in conjunction with Daniel Duan

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_set, val_set = random_split(train_dataset, [50000, 10000])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define modified CNN architecture
import torch.nn.functional as F

class ModifiedCNN(nn.Module):
    def __init__(self, num_channels):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, 3)
        self.conv2 = nn.Conv2d(num_channels, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 1 * 1, 512)  # Adjusted based on pooling and conv layer output size
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(100 * val_correct / val_total)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Validation Loss: {val_running_loss / len(val_loader):.4f}, '
              f'Train Acc: {100 * correct / total:.2f}%, Val Acc: {100 * val_correct / val_total:.2f}%')
    return train_loss, val_loss, train_acc, val_acc

# Find the best number of channels for the first convolutional layer
num_channels_list = [16, 32, 64]
best_val_acc = 0
best_num_channels = num_channels_list[0]

for num_channels in num_channels_list:
    print(f"Training with {num_channels} channels in the first convolutional layer.")
    model = ModifiedCNN(num_channels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    _, _, _, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    if max(val_acc) > best_val_acc:
        best_val_acc = max(val_acc)
        best_num_channels = num_channels

print(f"Best number of channels for the first convolutional layer: {best_num_channels} with validation accuracy: {best_val_acc:.2f}%")

# Create model with the best number of channels
model = ModifiedCNN(best_num_channels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer)

# Plot training & validation accuracy/loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()

# Evaluate the model on test set
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')

# Visualizing Feature Extraction
def plot_mapped_features(model, image, layers):
    x = image.unsqueeze(0)
    num_layers = len(layers)
    
    fig, axes = plt.subplots(num_layers, x.shape[1], figsize=(15, num_layers*2))
    
    for i, layer in enumerate(layers):
        x = layer(x)
        for j in range(x.shape[1]):
            ax = axes[i] if num_layers > 1 else axes
            ax.imshow(x[0, j].detach().cpu().numpy(), cmap='gray')
            ax.axis('off')
    
    plt.show()

examples = iter(test_loader)
example_data, example_labels = next(examples)
example_image = example_data[0]
layers = [model.conv1, model.pool, model.conv2, model.pool, model.conv3, model.pool]
plot_mapped_features(model, example_image, layers)

# Visualizing and Interpreting Filters
def plot_filters(layer, n_filters=6):
    filters = layer.weight.data.clone()
    filters = filters - filters.min()
    filters = filters / filters.max()
    filters = filters[:n_filters]
    fig, axes = plt.subplots(1, n_filters, figsize=(15, 5))
    for i, filter in enumerate(filters):
        axes[i].imshow(np.transpose(filter, (1, 2, 0)), cmap='gray')
        axes[i].axis('off')
    plt.show()

plot_filters(model.conv1, n_filters=6)

# Logistic Regression comparison
# Flattening the images
train_flat = train_dataset.data.view(-1, 28*28).numpy()
train_labels = train_dataset.targets.numpy()
test_flat = test_dataset.data.view(-1, 28*28).numpy()
test_labels = test_dataset.targets.numpy()

# Training logistic regression model
logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(train_flat, train_labels)

# Predicting and evaluating logistic regression
logistic_preds = logisticRegr.predict(test_flat)
logistic_acc = accuracy_score(test_labels, logistic_preds)
print(f'Logistic Regression Test Accuracy: {logistic_acc:.2f}%')

# Using hidden states of the CNN as inputs for logistic regression
def extract_hidden_states(model, loader):
    model.eval()
    hidden_states = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            x = model.pool(torch.relu(model.conv1(images)))
            x = model.pool(torch.relu(model.conv2(x)))
            x = model.pool(torch.relu(model.conv3(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            hidden_states.extend(x.numpy())
            labels.extend(lbls.numpy())
    return np.array(hidden_states), np.array(labels)

train_hidden, train_hidden_labels = extract_hidden_states(model, train_loader)
test_hidden, test_hidden_labels = extract_hidden_states(model, test_loader)

# Logistic Regression on hidden states
logisticRegr_hidden = LogisticRegression(max_iter=1000)
logisticRegr_hidden.fit(train_hidden, train_hidden_labels)

# Predicting and evaluating logistic regression on hidden states
logistic_preds_hidden = logisticRegr_hidden.predict(test_hidden)
logistic_hidden_acc = accuracy_score(test_hidden_labels, logistic_preds_hidden)
print(f'Logistic Regression on CNN Hidden States Test Accuracy: {logistic_hidden_acc:.2f}%')
