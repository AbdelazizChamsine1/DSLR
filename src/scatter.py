import sys
import os
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv, get_numerical_columns

NON_COURSE_COLUMNS = {'Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'}

KNOWN_HOUSE_COLORS = {
    'Gryffindor': '#c0392b',
    'Hufflepuff': '#f1c40f',
    'Ravenclaw':  '#2980b9',
    'Slytherin':  '#27ae60',
}


def extract_houses(dataset: dict) -> list:
    seen = []
    for h in dataset.get('Hogwarts House', []):
        h = h.strip()
        if h and h not in seen:
            seen.append(h)
    return sorted(seen)


def extract_courses(dataset: dict) -> list:
    numerical = get_numerical_columns(dataset)
    return [col for col in numerical if col not in NON_COURSE_COLUMNS]


def build_house_colors(houses: list) -> dict:
    import matplotlib.cm as cm
    palette = cm.get_cmap('tab10')
    colors = {}
    fallback_idx = 0
    for house in houses:
        if house in KNOWN_HOUSE_COLORS:
            colors[house] = KNOWN_HOUSE_COLORS[house]
        else:
            colors[house] = palette(fallback_idx)
            fallback_idx += 1
    return colors


def pearson_correlation(x: list, y: list) -> float:
    """Compute Pearson correlation between two lists of equal length."""
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((v - mean_x) ** 2 for v in x))
    den_y = math.sqrt(sum((v - mean_y) ** 2 for v in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def get_paired_values(col_a: list, col_b: list) -> tuple:
    """Return two aligned lists of floats, skipping rows where either value is missing."""
    out_a, out_b = [], []
    for a, b in zip(col_a, col_b):
        try:
            fa, fb = float(a), float(b)
            if not math.isnan(fa) and not math.isnan(fb):
                out_a.append(fa)
                out_b.append(fb)
        except (ValueError, TypeError):
            continue
    return out_a, out_b


def scatter_plot(filepath: str):
    dataset = parse_csv(filepath)
    courses = extract_courses(dataset)
    houses = extract_houses(dataset)
    house_colors = build_house_colors(houses)
    house_col = dataset.get('Hogwarts House', [])

    # Find the two most correlated features (highest |r|, excluding self-pairs)
    best_r = -1.0
    best_pair = (courses[0], courses[1])

    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            x, y = get_paired_values(dataset[courses[i]], dataset[courses[j]])
            r = abs(pearson_correlation(x, y))
            if r > best_r:
                best_r = r
                best_pair = (courses[i], courses[j])

    feat_a, feat_b = best_pair
    print(f'Most similar features: "{feat_a}" & "{feat_b}" (|r| = {best_r:.4f})')

    # Build per-house point lists for the best pair
    col_a = dataset[feat_a]
    col_b = dataset[feat_b]

    fig, ax = plt.subplots(figsize=(9, 6))

    for house in houses:
        xs, ys = [], []
        for h, a, b in zip(house_col, col_a, col_b):
            if h.strip() != house:
                continue
            try:
                fa, fb = float(a), float(b)
                if not math.isnan(fa) and not math.isnan(fb):
                    xs.append(fa)
                    ys.append(fb)
            except (ValueError, TypeError):
                continue
        ax.scatter(xs, ys, label=house, color=house_colors[house],
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
