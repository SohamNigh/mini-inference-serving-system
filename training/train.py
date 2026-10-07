"""Train 784-128-10 using only NumPy. Input: MNIST-format IDX gzip files."""
import argparse
import gzip
import json
import struct
from pathlib import Path
import numpy as np
from weights_io import export

def load(images_path, labels_path):
    with gzip.open(images_path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError("expected MNIST images")
        x = np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, 784).astype(np.float32) / 255
    with gzip.open(labels_path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError("expected MNIST labels")
        y = np.frombuffer(stream.read(), dtype=np.uint8)
        if len(y) != count or len(x) != count or np.any(y > 9):
            raise ValueError("invalid labels")
    return x, y

def forward(x, w1, b1, w2, b2):
    h = np.maximum(x @ w1 + b1, 0)
    logits = h @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return h, probabilities

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "weights.bin")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    x, y = load(args.data / "train-images-idx3-ubyte.gz", args.data / "train-labels-idx1-ubyte.gz")
    test_x, test_y = load(args.data / "t10k-images-idx3-ubyte.gz", args.data / "t10k-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(42)
    w1 = (rng.standard_normal((784, 128)) * np.sqrt(2 / 784)).astype(np.float32)
    w2 = (rng.standard_normal((128, 10)) * np.sqrt(2 / 128)).astype(np.float32)
    b1, b2 = np.zeros(128, np.float32), np.zeros(10, np.float32)
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            indices = order[start:start + 128]
            inputs, labels = x[indices], y[indices]
            hidden, gradient = forward(inputs, w1, b1, w2, b2)
            gradient[np.arange(len(indices)), labels] -= 1
            gradient /= len(indices)
            hidden_gradient = (gradient @ w2.T) * (hidden > 0)
            # Compute all gradients against the old weights before updating.
            dw1, dw2 = inputs.T @ hidden_gradient, hidden.T @ gradient
            w1 -= 0.1 * dw1
            b1 -= 0.1 * hidden_gradient.sum(axis=0)
            w2 -= 0.1 * dw2
            b2 -= 0.1 * gradient.sum(axis=0)
        _, probabilities = forward(test_x, w1, b1, w2, b2)
        print(f"epoch {epoch + 1}: test accuracy {np.mean(probabilities.argmax(axis=1) == test_y):.4f}")
    export(args.output, w1, b1, w2, b2)
    sample = Path(__file__).parent.parent / "testdata/sample_digit.json"
    sample.write_text(json.dumps({"pixels": test_x[0].tolist(), "label": int(test_y[0])}))

if __name__ == "__main__":
    main()
