import sys
import os
import math
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv, filter_nan
from utils.stat_utils import get_mean, get_std, sigmoid, dot
from utils.plot_utils import HOUSES

FEATURES = [
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Charms',
    'Flying',
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: str):
    dataset = parse_csv(filepath)
    house_col = dataset.get('Hogwarts House', [])
    n_rows = len(house_col)

    # Compute mean and std for each feature (on non-NaN values)
    means = {}
    stds = {}
    for feat in FEATURES:
        col = dataset.get(feat, [])
        clean = filter_nan(col)
        means[feat] = get_mean(clean)
        stds[feat] = get_std(clean)

    # Build X (list of feature vectors) and houses list
    X = []
    y_labels = []

    for i in range(n_rows):
        house = house_col[i].strip() if i < len(house_col) else ''
        if house not in HOUSES:
            continue

        row = []
        valid = True
        for feat in FEATURES:
            col = dataset.get(feat, [])
            raw = col[i] if i < len(col) else ''
            try:
                val = float(raw)
                if math.isnan(val):
                    val = means[feat]   # impute with mean
            except (ValueError, TypeError):
                val = means[feat]       # impute with mean
            std = stds[feat] if stds[feat] != 0 else 1.0
            row.append((val - means[feat]) / std)

        X.append(row)
        y_labels.append(house)

    return X, y_labels, means, stds


# ---------------------------------------------------------------------------
# Training — one-vs-all logistic regression
# ---------------------------------------------------------------------------

def train_one_vs_all(X: list, y_labels: list, house: str,
                     lr: float = 0.1, epochs: int = 1000) -> list:
    n_samples = len(X)
    n_features = len(X[0])

    # weights[0] = bias, weights[1:] = feature weights
    weights = [0.0] * (n_features + 1)

    y = [1.0 if label == house else 0.0 for label in y_labels]

    for epoch in range(epochs):
        grad = [0.0] * (n_features + 1)

        for i in range(n_samples):
            z = weights[0] + dot(weights[1:], X[i])
            h = sigmoid(z)
            error = h - y[i]

            grad[0] += error
            for j in range(n_features):
                grad[j + 1] += error * X[i][j]

        for j in range(n_features + 1):
            weights[j] -= lr * grad[j] / n_samples

    return weights


def train(filepath: str, output: str = 'weights.json',
          lr: float = 0.1, epochs: int = 1000):
    print(f'Loading dataset: {filepath}')
    X, y_labels, means, stds = load_dataset(filepath)
    print(f'  Samples: {len(X)}  |  Features: {len(FEATURES)}')

    model = {
        'features': FEATURES,
        'means':    means,
        'stds':     stds,
        'weights':  {},
    }

    for house in HOUSES:
        print(f'  Training classifier: {house} vs all ...')
        w = train_one_vs_all(X, y_labels, house, lr=lr, epochs=epochs)
        model['weights'][house] = w
        print(f'    Done.')

    with open(output, 'w') as f:
        json.dump(model, f)
    print(f'Weights saved to {output}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python logreg_train.py <dataset_train.csv> [weights.json] [lr] [epochs]')
        sys.exit(1)

    filepath = sys.argv[1]
    output   = sys.argv[2] if len(sys.argv) > 2 else 'weights.json'
    lr       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    epochs   = int(sys.argv[4])   if len(sys.argv) > 4 else 1000

    train(filepath, output, lr, epochs)
