#!/usr/bin/env python3

import sys, os, socket, struct, csv, gzip
import numpy as np

MSG_REGISTER = 1
MSG_WEIGHTS  = 2
MSG_AVG      = 3

if len(sys.argv) != 7:
    print(f"usage: {sys.argv[0]} id host port shard rounds mnist_dir")
    print("  shard: e.g. 0-4, 5-9, or all")
    sys.exit(1)

wid       = sys.argv[1]
host      = sys.argv[2]
port      = int(sys.argv[3])
shard     = sys.argv[4]
rounds    = int(sys.argv[5])
data_dir  = sys.argv[6]

# --- load MNIST from raw IDX files ---
def read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n     = int.from_bytes(f.read(4), "big")
        if magic == 2051:
            r = int.from_bytes(f.read(4), "big")
            c = int.from_bytes(f.read(4), "big")
            buf = f.read(n * r * c)
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, r*c).astype(np.float32) / 255.0
        elif magic == 2049:
            return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)
        else:
            raise ValueError(f"bad magic {magic}")

def find(d, stem):
    for ext in ("", ".gz"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(stem)

X_train = read_idx(find(data_dir, "train-images-idx3-ubyte"))
y_train = read_idx(find(data_dir, "train-labels-idx1-ubyte"))
X_test  = read_idx(find(data_dir, "t10k-images-idx3-ubyte"))
y_test  = read_idx(find(data_dir, "t10k-labels-idx1-ubyte"))

# pick the classes we're allowed to train on
if shard == "all":
    keep = set(range(10))
else:
    lo, hi = map(int, shard.split("-"))
    keep = set(range(lo, hi + 1))

mask = np.isin(y_train, list(keep))
X = X_train[mask]
y = y_train[mask]
print(f"[{wid}] classes {sorted(keep)}, {len(X)} samples")

# --- model: 784 -> 128 -> 64 -> 10 ---
rng = np.random.RandomState(42 + hash(wid) % 1000)

def xavier(fin, fout):
    lim = np.sqrt(6.0 / (fin + fout))
    return rng.uniform(-lim, lim, (fout, fin)).astype(np.float32)

W1 = xavier(784, 128); b1 = np.zeros(128, dtype=np.float32)
W2 = xavier(128, 64);  b2 = np.zeros(64,  dtype=np.float32)
W3 = xavier(64,  10);  b3 = np.zeros(10,  dtype=np.float32)

def forward(Xb):
    z1 = Xb @ W1.T + b1; a1 = np.maximum(z1, 0)
    z2 = a1 @ W2.T + b2; a2 = np.maximum(z2, 0)
    z3 = a2 @ W3.T + b3
    # softmax
    s = z3 - z3.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    return p, (Xb, z1, a1, z2, a2, z3, p)

def backward(cache, yb, lr):
    global W1, b1, W2, b2, W3, b3
    Xb, z1, a1, z2, a2, z3, p = cache
    N = Xb.shape[0]
    Y = np.zeros_like(p); Y[np.arange(N), yb] = 1.0
    dz3 = (p - Y) / N
    dW3 = dz3.T @ a2;  db3 = dz3.sum(0)
    da2 = dz3 @ W3;    dz2 = da2 * (z2 > 0)
    dW2 = dz2.T @ a1;  db2 = dz2.sum(0)
    da1 = dz2 @ W2;    dz1 = da1 * (z1 > 0)
    dW1 = dz1.T @ Xb;  db1 = dz1.sum(0)
    W3 -= lr * dW3; b3 -= lr * db3
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

def train_epoch(bs=128, lr=1e-2):
    idx = rng.permutation(len(X))
    tl = 0.0; nb = 0
    for i in range(0, len(idx), bs):
        bi = idx[i:i+bs]
        Xb = X[bi]; yb = y[bi]
        p, c = forward(Xb)
        loss = -np.log(p[np.arange(len(yb)), yb] + 1e-12).mean()
        backward(c, yb, lr)
        tl += loss; nb += 1
    return tl / max(1, nb)

def evaluate():
    p, _ = forward(X_test)
    return 100.0 * (p.argmax(1) == y_test).mean()

# --- weight serialization, same layout as model.cpp ---
def to_floats():
    return np.concatenate([W1.flatten(), b1, W2.flatten(), b2, W3.flatten(), b3]).astype(np.float32)

def from_floats(flat):
    global W1, b1, W2, b2, W3, b3
    o = 0
    def take(shape):
        nonlocal o
        n = int(np.prod(shape))
        a = flat[o:o+n].reshape(shape).astype(np.float32).copy()
        o += n
        return a
    W1 = take((128, 784)); b1 = take((128,))
    W2 = take((64, 128));  b2 = take((64,))
    W3 = take((10, 64));   b3 = take((10,))

NFLOATS = (128*784) + 128 + (64*128) + 64 + (10*64) + 10

# --- networking ---
def recv_all(s, n):
    buf = bytearray()
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c: raise RuntimeError("socket closed")
        buf.extend(c)
    return bytes(buf)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
idb = wid.encode()
s.sendall(struct.pack("<ii", MSG_REGISTER, len(idb)) + idb)
print(f"[{wid}] connected to {host}:{port}")

# --- main loop ---
os.makedirs("results", exist_ok=True)
csv_path = f"results/accuracy_{wid}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "local_accuracy", "federated_accuracy"])

    for r in range(1, rounds + 1):
        loss = train_epoch()
        local_acc = evaluate()

        flat = to_floats()
        s.sendall(struct.pack("<ii", MSG_WEIGHTS, NFLOATS) + flat.tobytes())

        hdr = recv_all(s, 8)
        mt, nf = struct.unpack("<ii", hdr)
        if mt != MSG_AVG or nf != NFLOATS:
            print(f"[{wid}] bad response mt={mt}")
            break
        avg = np.frombuffer(recv_all(s, NFLOATS * 4), dtype=np.float32)
        from_floats(avg)

        fed_acc = evaluate()
        print(f"[{wid}] round {r:2d}  loss={loss:.4f}  local={local_acc:.2f}%  fed={fed_acc:.2f}%")
        w.writerow([r, f"{local_acc:.4f}", f"{fed_acc:.4f}"])
        f.flush()

s.close()

# save final weights so they can be used for inference
with open(f"results/weights_fed_{wid}.bin", "wb") as wf:
    wf.write(to_floats().tobytes())
print(f"[{wid}] saved results/weights_fed_{wid}.bin")