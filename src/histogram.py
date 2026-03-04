import sys
import os
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv
from utils.plot_utils import (
    HOUSES, HOUSE_COLORS,
    group_by_house, homogeneity_score,
)

COURSES = [
    'Arithmancy', 'Astronomy', 'Herbology',
    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
    'Ancient Runes', 'History of Magic', 'Transfiguration',
    'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
]


def histogram(filepath: str):
    dataset = parse_csv(filepath)

    scores = {c: homogeneity_score(group_by_house(dataset, c)) for c in COURSES}
    most_homogeneous = min(scores, key=scores.get)
    print(f'Most homogeneous course: {most_homogeneous} (score: {scores[most_homogeneous]:.4f})')

    cols = 3
    rows = math.ceil(len(COURSES) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    fig.suptitle(
        'Score distributions per Hogwarts House\n'
        f'→ Most homogeneous: {most_homogeneous}',
        fontsize=14, fontweight='bold'
    )

    for idx, course in enumerate(COURSES):
        ax = axes[idx // cols][idx % cols]
        groups = group_by_house(dataset, course)

        for house in HOUSES:
            data = groups[house]
            if not data:
                continue
            ax.hist(data, bins=20, alpha=0.5, label=house,
                    color=HOUSE_COLORS[house], edgecolor='none')

        ax.set_title(course, fontsize=9,
                     fontweight='bold' if course == most_homogeneous else 'normal')
        ax.set_xlabel('Score', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=7)

        if course == most_homogeneous:
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(2.5)

    for idx in range(len(COURSES), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HOUSE_COLORS[h], alpha=0.7, label=h)
        for h in HOUSES
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=10, title='House')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python histogram.py <dataset.csv>')
        sys.exit(1)
    histogram(sys.argv[1])
