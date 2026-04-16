#!/usr/bin/env python3
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


tfm = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=1000, shuffle=False)

model = MLP()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("training...")
for epoch in range(5):
    model.train()
    total = 0.0
    for x, y in train_dl:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"  epoch {epoch+1}/5  avg loss {total/len(train_dl):.4f}")

# test accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_dl:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"test accuracy: {100*correct/total:.2f}%")

# dump weights in the order model.cpp expects
def dump(f, t):
    flat = t.detach().cpu().numpy().astype("float32").flatten()
    for v in flat:
        f.write(struct.pack("f", v))

with open("weights.bin", "wb") as f:
    dump(f, model.fc1.weight); dump(f, model.fc1.bias)
    dump(f, model.fc2.weight); dump(f, model.fc2.bias)
    dump(f, model.fc3.weight); dump(f, model.fc3.bias)
print("wrote weights.bin")

# dump test set for the C++ client
# format: int32 count, then for each sample: 784 float32 pixels, int32 label
n = len(test_ds)
with open("mnist_test.bin", "wb") as f:
    f.write(struct.pack("i", n))
    for img, label in test_ds:
        px = img.numpy().astype("float32").flatten()
        for v in px:
            f.write(struct.pack("f", v))
        f.write(struct.pack("i", int(label)))
print(f"wrote mnist_test.bin ({n} samples)")
