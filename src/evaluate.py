import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv
from logreg_train import load_dataset, train_one_vs_all, FEATURES
from utils.stat_utils import sigmoid, dot
from utils.plot_utils import HOUSES


def train_val_split(X: list, y: list, val_ratio: float = 0.2, seed: int = 42):
    n = len(X)
    # Deterministic shuffle using a simple LCG
    indices = list(range(n))
    a, c, m = 1664525, 1013904223, 2 ** 32
    rng = seed
    for i in range(n - 1, 0, -1):
        rng = (a * rng + c) % m
        j = rng % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    split = int(n * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val   = [X[i] for i in val_idx]
    y_val   = [y[i] for i in val_idx]

    return X_train, y_train, X_val, y_val


def predict_sample(x: list, weights: dict) -> str:
    best_house = None
    best_prob  = -1.0
    for house, w in weights.items():
        z    = w[0] + dot(w[1:], x)
        prob = sigmoid(z)
        if prob > best_prob:
            best_prob  = prob
            best_house = house
    return best_house


def evaluate(filepath: str, val_ratio: float = 0.2,
             lr: float = 0.1, epochs: int = 1000):
    print(f'Loading dataset: {filepath}')
    X, y_labels, means, stds = load_dataset(filepath)
    print(f'  Total samples: {len(X)}')

    X_train, y_train, X_val, y_val = train_val_split(X, y_labels, val_ratio)
    print(f'  Train: {len(X_train)}  |  Val: {len(X_val)}')

    # Train one-vs-all classifiers
    weights = {}
    for house in HOUSES:
        print(f'  Training: {house} vs all ...')
        weights[house] = train_one_vs_all(X_train, y_train, house, lr=lr, epochs=epochs)

    # Evaluate on validation set
    correct = 0
    per_house = {h: {'tp': 0, 'total': 0} for h in HOUSES}

    for x, true_label in zip(X_val, y_val):
        pred = predict_sample(x, weights)
        per_house[true_label]['total'] += 1
        if pred == true_label:
            correct += 1
            per_house[true_label]['tp'] += 1

    accuracy = correct / len(y_val) * 100
    print(f'\n--- Results ({int(val_ratio * 100)}% validation set) ---')
    print(f'Overall accuracy: {correct}/{len(y_val)}  =  {accuracy:.2f}%')
    print()
    print(f'{"House":<12}  {"Correct":>7}  {"Total":>7}  {"Acc":>7}')
    print('-' * 40)
    for house in HOUSES:
        tp    = per_house[house]['tp']
        total = per_house[house]['total']
        acc   = (tp / total * 100) if total > 0 else 0.0
        print(f'{house:<12}  {tp:>7}  {total:>7}  {acc:>6.2f}%')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python evaluate.py <dataset_train.csv> [val_ratio] [lr] [epochs]')
        sys.exit(1)

    filepath  = sys.argv[1]
    val_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    lr        = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    epochs    = int(sys.argv[4])   if len(sys.argv) > 4 else 1000

    evaluate(filepath, val_ratio, lr, epochs)
