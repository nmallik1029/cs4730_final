#!/usr/bin/env python3
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# MODEL
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load MNIST 
transform = transforms.Compose([
    transforms.ToTensor(),  # values in [0,1]
])

train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)


# TRAINING
device = torch.device("cpu")
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training MLP on MNIST")
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS}  avg loss: {total_loss/len(train_loader):.4f}")

# EVAL
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
print(f"Test accuracy: {100.0 * correct / total:.2f}%")


def write_floats(f, tensor):
    flat = tensor.detach().cpu().numpy().astype("float32").flatten()
    for v in flat:
        f.write(struct.pack("f", v))

with open("weights.bin", "wb") as f:
    write_floats(f, model.fc1.weight)  # (128, 784)
    write_floats(f, model.fc1.bias)    # (128,)
    write_floats(f, model.fc2.weight)  # (64, 128)
    write_floats(f, model.fc2.bias)    # (64,)
    write_floats(f, model.fc3.weight)  # (10, 64)
    write_floats(f, model.fc3.bias)    # (10,)

print("Wrote weights.bin")


num_test = len(test_set)
with open("mnist_test.bin", "wb") as f:
    f.write(struct.pack("i", num_test))
    for img, label in test_set:
        arr = img.numpy().astype("float32").flatten()  # 784 floats
        for v in arr:
            f.write(struct.pack("f", v))
        f.write(struct.pack("i", int(label)))

print(f"Wrote mnist_test.bin with {num_test} images")
