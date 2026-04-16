#!/usr/bin/env python3
"""
train_worker.py
Federated learning worker. Trains a local MLP on its shard of MNIST,
then participates in R rounds of federated averaging with the coordinator.

Usage:
    python train_worker.py <worker_id> <coord_host> <coord_port> <shard_spec> <rounds>

Example:
    # Worker A handles digits 0-4
    python train_worker.py A localhost 6000 0-4 20

    # Worker B handles digits 5-9
    python train_worker.py B localhost 6000 5-9 20

Also logs accuracy at each round to accuracy_<worker_id>.csv
so we can plot convergence after training.

Protocol (all little-endian):
    Handshake (worker -> coord):
        int32  MSG_REGISTER (=1)
        int32  worker_id_len
        bytes  worker_id (utf-8)

    Each round:
        worker -> coord:
            int32  MSG_WEIGHTS (=2)
            int32  num_floats
            float32[num_floats]  weights (W1,b1,W2,b2,W3,b3 flat)

        coord -> worker:
            int32  MSG_AVG (=3)
            int32  num_floats
            float32[num_floats]  averaged weights
"""

import sys
import socket
import struct
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

MSG_REGISTER = 1
MSG_WEIGHTS  = 2
MSG_AVG      = 3

# ----- Args -----
if len(sys.argv) != 6:
    print(f"Usage: {sys.argv[0]} <worker_id> <coord_host> <coord_port> <shard_spec> <rounds>")
    print("  shard_spec examples: '0-4' (digits 0,1,2,3,4)  or  '5-9'  or  'all'")
    sys.exit(1)

worker_id  = sys.argv[1]
coord_host = sys.argv[2]
coord_port = int(sys.argv[3])
shard_spec = sys.argv[4]
rounds     = int(sys.argv[5])

# ----- Parse shard -----
if shard_spec == "all":
    allowed_classes = list(range(10))
else:
    lo, hi = shard_spec.split("-")
    allowed_classes = list(range(int(lo), int(hi) + 1))
print(f"[{worker_id}] training on classes {allowed_classes}")

# ----- Load MNIST shard -----
transform = transforms.ToTensor()
train_full = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_full  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_idx = [i for i, (_, y) in enumerate(train_full) if y in allowed_classes]
train_set = Subset(train_full, train_idx)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

# Evaluate against full test set each round (critical!)
# -- this is what lets us show the "worker-alone is stuck at 50%" story
test_loader = DataLoader(test_full, batch_size=1000, shuffle=False)
print(f"[{worker_id}] local shard: {len(train_set)} training samples")

# ----- Model -----
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

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----- Weight (de)serialization -----
# Same layout as weights_export.py / model.cpp:
#   W1 (128x784), b1 (128), W2 (64x128), b2 (64), W3 (10x64), b3 (10)
def model_to_floats():
    parts = [model.fc1.weight, model.fc1.bias,
             model.fc2.weight, model.fc2.bias,
             model.fc3.weight, model.fc3.bias]
    return np.concatenate([p.detach().cpu().numpy().astype(np.float32).flatten() for p in parts])

def floats_to_model(flat):
    offset = 0
    def take(shape):
        nonlocal offset
        n = int(np.prod(shape))
        arr = flat[offset:offset+n].reshape(shape)
        offset += n
        return torch.tensor(arr, dtype=torch.float32)
    with torch.no_grad():
        model.fc1.weight.copy_(take((128, 784)))
        model.fc1.bias.copy_(take((128,)))
        model.fc2.weight.copy_(take((64, 128)))
        model.fc2.bias.copy_(take((64,)))
        model.fc3.weight.copy_(take((10, 64)))
        model.fc3.bias.copy_(take((10,)))

NUM_FLOATS = (128*784) + 128 + (64*128) + 64 + (10*64) + 10

# ----- Networking -----
def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed prematurely")
        buf.extend(chunk)
    return bytes(buf)

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((coord_host, coord_port))
    # Register
    payload = worker_id.encode("utf-8")
    s.sendall(struct.pack("<ii", MSG_REGISTER, len(payload)) + payload)
    return s

# ----- Evaluate -----
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# ----- Main loop -----
print(f"[{worker_id}] connecting to coordinator {coord_host}:{coord_port}")
sock = connect()
print(f"[{worker_id}] registered, starting training")

csv_path = f"accuracy_{worker_id}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "local_accuracy", "federated_accuracy"])

    for r in range(1, rounds + 1):
        # --- Local training for 1 epoch ---
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Accuracy AFTER local training, BEFORE federated averaging
        local_acc = evaluate()

        # --- Send weights to coordinator ---
        flat = model_to_floats()
        sock.sendall(struct.pack("<ii", MSG_WEIGHTS, NUM_FLOATS) + flat.tobytes())

        # --- Receive averaged weights back ---
        header = recv_exact(sock, 8)
        mtype, nf = struct.unpack("<ii", header)
        if mtype != MSG_AVG or nf != NUM_FLOATS:
            print(f"[{worker_id}] unexpected message: type={mtype} nf={nf}")
            break
        data = recv_exact(sock, NUM_FLOATS * 4)
        avg = np.frombuffer(data, dtype=np.float32)
        floats_to_model(avg)

        # Accuracy AFTER federated averaging
        fed_acc = evaluate()

        print(f"[{worker_id}] round {r:2d}  loss={total_loss/len(train_loader):.4f}"
              f"  local_acc={local_acc:.2f}%  fed_acc={fed_acc:.2f}%")
        writer.writerow([r, f"{local_acc:.4f}", f"{fed_acc:.4f}"])
        f.flush()

sock.close()

# ----- Save final weights for inference -----
# (Same format as weights_export.py — can feed into worker.cpp)
out_path = f"weights_fed_{worker_id}.bin"
with open(out_path, "wb") as wf:
    wf.write(model_to_floats().tobytes())
print(f"[{worker_id}] wrote final weights to {out_path}")
