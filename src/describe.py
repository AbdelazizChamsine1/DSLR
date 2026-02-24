import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv, get_numerical_columns
from utils.stat_utils import (
    get_count, get_mean, get_std,
    get_min, get_max, get_percentile
)


def describe(filepath: str):
    dataset = parse_csv(filepath)
    numerical = get_numerical_columns(dataset)

    features = list(numerical.keys())

    rows = {
        'Count': [],
        'Mean':  [],
        'Std':   [],
        'Min':   [],
        '25%':   [],
        '50%':   [],
        '75%':   [],
        'Max':   [],
    }

    for feature in features:
        data = numerical[feature]
        rows['Count'].append(get_count(data))
        rows['Mean'].append(get_mean(data))
        rows['Std'].append(get_std(data))
        rows['Min'].append(get_min(data))
        rows['25%'].append(get_percentile(data, 25))
        rows['50%'].append(get_percentile(data, 50))
        rows['75%'].append(get_percentile(data, 75))
        rows['Max'].append(get_max(data))

    # --- formatting ---
    col_width = 16
    label_width = 8

    # truncate long feature names for display
    headers = [f[:col_width - 1] for f in features]

    header_line = ' ' * label_width + ''.join(h.rjust(col_width) for h in headers)
    print(header_line)

    for stat, values in rows.items():
        line = stat.ljust(label_width)
        for v in values:
            formatted = f'{v:.6f}'
            line += formatted.rjust(col_width)
        print(line)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python describe.py <dataset.csv>')
        sys.exit(1)
    describe(sys.argv[1])
