#!/usr/bin/env python3
import os
import urllib.request

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

dst = "./data/MNIST/raw"
os.makedirs(dst, exist_ok=True)

for fn in FILES:
    out = os.path.join(dst, fn)
    if os.path.exists(out):
        print(f"have {fn}")
        continue
    for base in MIRRORS:
        try:
            print(f"fetching {base+fn}")
            urllib.request.urlretrieve(base + fn, out)
            break
        except Exception as e:
            print(f"  fail: {e}")
    else:
        raise SystemExit(f"couldn't get {fn}")

print("done.")
