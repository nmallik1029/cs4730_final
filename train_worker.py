#!/usr/bin/env python3
import sys
import os
import socket
import struct
import csv
import gzip
import numpy as np

MSG_REGISTER = 1
MSG_WEIGHTS  = 2
MSG_AVG      = 3

# ---------------- Args ----------------
if len(sys.argv) != 7:
    print(f"Usage: {sys.argv[0]} <worker_id> <coord_host> <coord_port> "
          f"<shard_spec> <rounds> <mnist_data_dir>")
    sys.exit(1)

worker_id   = sys.argv[1]
coord_host  = sys.argv[2]
coord_port  = int(sys.argv[3])
shard_spec  = sys.argv[4]
rounds      = int(sys.argv[5])
data_dir    = sys.argv[6]

def _read_idx(path):
    """Read an IDX file (optionally gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic == 2051:  # images
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            buf = f.read(n * rows * cols)
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0
        elif magic == 2049:  # labels
            buf = f.read(n)
            return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        else:
            raise ValueError(f"Unknown IDX magic {magic} in {path}")

def _find(base, stem):
    """Find 'stem' or 'stem.gz' in base dir."""
    for ext in ("", ".gz"):
        p = os.path.join(base, stem + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{stem} not found in {base} (tried .gz too)")

X_train = _read_idx(_find(data_dir, "train-images-idx3-ubyte"))
y_train = _read_idx(_find(data_dir, "train-labels-idx1-ubyte"))
X_test  = _read_idx(_find(data_dir, "t10k-images-idx3-ubyte"))
y_test  = _read_idx(_find(data_dir, "t10k-labels-idx1-ubyte"))

# ---------------- Shard ----------------
if shard_spec == "all":
    allowed = set(range(10))
else:
    lo, hi = map(int, shard_spec.split("-"))
    allowed = set(range(lo, hi + 1))

mask = np.isin(y_train, list(allowed))
X_shard = X_train[mask]
y_shard = y_train[mask]
print(f"[{worker_id}] classes {sorted(allowed)} - {len(X_shard)} training samples")

# ---------------- Model (pure numpy) ----------------
# Architecture: 784 -> 128 -> 64 -> 10  (same as model.cpp and weights_export.py)
rng = np.random.RandomState(42 + hash(worker_id) % 1000)

def xavier(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)

W1 = xavier(784, 128);  b1 = np.zeros(128, dtype=np.float32)
W2 = xavier(128, 64);   b2 = np.zeros(64,  dtype=np.float32)
W3 = xavier(64,  10);   b3 = np.zeros(10,  dtype=np.float32)

def forward(X):
    z1 = X @ W1.T + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2.T + b2
    a2 = np.maximum(z2, 0)
    z3 = a2 @ W3.T + b3
    # softmax for cross-entropy
    shift = z3 - z3.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    p = exp / exp.sum(axis=1, keepdims=True)
    cache = (X, z1, a1, z2, a2, z3, p)
    return p, cache

def backward(cache, y_true, lr):
    global W1, b1, W2, b2, W3, b3
    X, z1, a1, z2, a2, z3, p = cache
    N = X.shape[0]
    # one-hot
    Y = np.zeros_like(p); Y[np.arange(N), y_true] = 1.0
    # gradients
    dz3 = (p - Y) / N                        # (N, 10)
    dW3 = dz3.T @ a2                         # (10, 64)
    db3 = dz3.sum(axis=0)                    # (10,)
    da2 = dz3 @ W3                           # (N, 64)
    dz2 = da2 * (z2 > 0)
    dW2 = dz2.T @ a1                         # (64, 128)
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2
    dz1 = da1 * (z1 > 0)
    dW1 = dz1.T @ X                          # (128, 784)
    db1 = dz1.sum(axis=0)
    # update
    W3 -= lr * dW3; b3 -= lr * db3
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

def train_one_epoch(batch_size=128, lr=1e-2):
    idx = rng.permutation(len(X_shard))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i+batch_size]
        X = X_shard[bi]; y = y_shard[bi]
        p, cache = forward(X)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        backward(cache, y, lr)
        total_loss += loss
        n_batches += 1
    return total_loss / max(1, n_batches)

def evaluate():
    # Use full test set
    p, _ = forward(X_test)
    pred = p.argmax(axis=1)
    return 100.0 * (pred == y_test).mean()

# ---------------- Weight (de)serialization ----------------
# Order must match model.cpp: W1, b1, W2, b2, W3, b3, each row-major.
def model_to_floats():
    return np.concatenate([W1.flatten(), b1, W2.flatten(), b2, W3.flatten(), b3]).astype(np.float32)

def floats_to_model(flat):
    global W1, b1, W2, b2, W3, b3
    off = 0
    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        arr = flat[off:off+n].reshape(shape)
        off += n
        return arr.astype(np.float32).copy()
    W1 = take((128, 784)); b1 = take((128,))
    W2 = take((64, 128));  b2 = take((64,))
    W3 = take((10, 64));   b3 = take((10,))

NUM_FLOATS = (128*784) + 128 + (64*128) + 64 + (10*64) + 10

# ---------------- Networking ----------------
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
    payload = worker_id.encode("utf-8")
    s.sendall(struct.pack("<ii", MSG_REGISTER, len(payload)) + payload)
    return s

# ---------------- Main loop ----------------
print(f"[{worker_id}] connecting to coordinator {coord_host}:{coord_port}")
sock = connect()
print(f"[{worker_id}] registered, starting {rounds} rounds")

csv_path = f"accuracy_{worker_id}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "local_accuracy", "federated_accuracy"])

    for r in range(1, rounds + 1):
        loss = train_one_epoch()
        local_acc = evaluate()

        # Send weights
        flat = model_to_floats()
        sock.sendall(struct.pack("<ii", MSG_WEIGHTS, NUM_FLOATS) + flat.tobytes())

        # Receive averaged weights
        header = recv_exact(sock, 8)
        mtype, nf = struct.unpack("<ii", header)
        if mtype != MSG_AVG or nf != NUM_FLOATS:
            print(f"[{worker_id}] unexpected msg: type={mtype} nf={nf}")
            break
        data = recv_exact(sock, NUM_FLOATS * 4)
        avg = np.frombuffer(data, dtype=np.float32)
        floats_to_model(avg)

        fed_acc = evaluate()
        print(f"[{worker_id}] round {r:2d}  loss={loss:.4f}  local={local_acc:.2f}%  fed={fed_acc:.2f}%")
        writer.writerow([r, f"{local_acc:.4f}", f"{fed_acc:.4f}"])
        f.flush()

sock.close()

# Save final weights for inference
with open(f"weights_fed_{worker_id}.bin", "wb") as wf:
    wf.write(model_to_floats().tobytes())
print(f"[{worker_id}] wrote weights_fed_{worker_id}.bin")