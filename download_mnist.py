#!/usr/bin/env python3
"""
download_mnist.py
Downloads the 4 MNIST IDX files using only Python stdlib.
Saves to ./data/MNIST/raw/

Usage:
    python3 download_mnist.py
"""

import os
import urllib.request

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# Mirror list in case one is down
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

DEST = "./data/MNIST/raw"
os.makedirs(DEST, exist_ok=True)

for fname in FILES:
    out_path = os.path.join(DEST, fname)
    if os.path.exists(out_path):
        print(f"[skip] {fname} already exists")
        continue
    for base in MIRRORS:
        url = base + fname
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"  -> {out_path}")
            break
        except Exception as e:
            print(f"  failed: {e}")
    else:
        print(f"ERROR: could not download {fname} from any mirror")
        raise SystemExit(1)

print("\nAll MNIST files downloaded to", DEST)
