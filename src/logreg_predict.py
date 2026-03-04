import sys
import os
import math
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv
from utils.stat_utils import sigmoid, dot


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(dataset_path: str, weights_path: str, output: str = 'houses.csv'):
    # Load model
    with open(weights_path, 'r') as f:
        model = json.load(f)

    features = model['features']
    means    = model['means']
    stds     = model['stds']
    weights  = model['weights']   # {house: [bias, w1, w2, ...]}
    houses   = list(weights.keys())

    # Load test dataset
    dataset = parse_csv(dataset_path)
    index_col = dataset.get('Index', [])
    n_rows = len(index_col)

    predictions = []

    for i in range(n_rows):
        # Build normalized feature vector, impute missing with 0 (= mean after normalization)
        row = []
        for feat in features:
            col = dataset.get(feat, [])
            raw = col[i] if i < len(col) else ''
            try:
                val = float(raw)
                if math.isnan(val):
                    val = means[feat]
            except (ValueError, TypeError):
                val = means[feat]
            std = stds[feat] if stds[feat] != 0 else 1.0
            row.append((val - means[feat]) / std)

        # Compute probability for each house, pick the highest
        best_house = None
        best_prob  = -1.0
        for house in houses:
            w = weights[house]
            z = w[0] + dot(w[1:], row)
            prob = sigmoid(z)
            if prob > best_prob:
                best_prob  = prob
                best_house = house

        predictions.append((index_col[i], best_house))

    # Write output
    with open(output, 'w') as f:
        f.write('Index,Hogwarts House\n')
        for idx, house in predictions:
            f.write(f'{idx},{house}\n')

    print(f'Predictions saved to {output}  ({len(predictions)} rows)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python logreg_predict.py <dataset_test.csv> <weights.json> [houses.csv]')
        sys.exit(1)

    dataset_path  = sys.argv[1]
    weights_path  = sys.argv[2]
    output        = sys.argv[3] if len(sys.argv) > 3 else 'houses.csv'

    predict(dataset_path, weights_path, output)
