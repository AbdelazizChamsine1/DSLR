import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv
from utils.plot_utils import (
    HOUSE_COLORS,
    extract_houses, extract_courses,
    get_paired_values, pearson_correlation,
)


def scatter_plot(filepath: str):
    dataset = parse_csv(filepath)
    courses = extract_courses(dataset)
    houses  = extract_houses(dataset)
    house_col = dataset.get('Hogwarts House', [])

    # Find the two most correlated features
    best_r    = -1.0
    best_pair = (courses[0], courses[1])

    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            x, y = get_paired_values(dataset[courses[i]], dataset[courses[j]])
            r = abs(pearson_correlation(x, y))
            if r > best_r:
                best_r    = r
                best_pair = (courses[i], courses[j])

    feat_a, feat_b = best_pair
    print(f'Most similar features: "{feat_a}" & "{feat_b}" (|r| = {best_r:.4f})')

    col_a = dataset[feat_a]
    col_b = dataset[feat_b]

    fig, ax = plt.subplots(figsize=(9, 6))

    for house in houses:
        xs, ys = [], []
        for h, a, b in zip(house_col, col_a, col_b):
            if h.strip() != house:
                continue
            fa, fb = None, None
            try:
                import math
                fa, fb = float(a), float(b)
                if math.isnan(fa) or math.isnan(fb):
                    continue
            except (ValueError, TypeError):
                continue
            xs.append(fa)
            ys.append(fb)
        ax.scatter(xs, ys, label=house, color=HOUSE_COLORS.get(house, 'grey'),
                   alpha=0.6, s=18, edgecolors='none')

    ax.set_xlabel(feat_a, fontsize=11)
    ax.set_ylabel(feat_b, fontsize=11)
    ax.set_title(
        f'Most similar features\n"{feat_a}" vs "{feat_b}"  (|r| = {best_r:.4f})',
        fontsize=12, fontweight='bold'
    )
    ax.legend(title='House', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python scatter.py <dataset.csv>')
        sys.exit(1)
    scatter_plot(sys.argv[1])
