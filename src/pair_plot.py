import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import parse_csv
from utils.plot_utils import (
    HOUSE_COLORS,
    extract_houses, extract_courses, get_house_data,
)


def pair_plot(filepath: str):
    dataset = parse_csv(filepath)
    houses  = extract_houses(dataset)
    courses = extract_courses(dataset)

    house_data = get_house_data(dataset, courses, houses)

    n = len(courses)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    fig.suptitle('Pair Plot — Hogwarts student scores', fontsize=14, fontweight='bold')

    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            course_x = courses[col]
            course_y = courses[row]

            if row == col:
                for house in houses:
                    data = house_data[house][course_x]
                    if data:
                        ax.hist(data, bins=15, alpha=0.5,
                                color=HOUSE_COLORS.get(house, 'grey'),
                                edgecolor='none', density=True)
            else:
                for house in houses:
                    xs = house_data[house][course_x]
                    ys = house_data[house][course_y]
                    length = min(len(xs), len(ys))
                    if length:
                        ax.scatter(xs[:length], ys[:length],
                                   color=HOUSE_COLORS.get(house, 'grey'),
                                   alpha=0.3, s=3, edgecolors='none')

            if row == n - 1:
                ax.set_xlabel(course_x, fontsize=5, rotation=15, ha='right')
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(course_y, fontsize=5, rotation=0, ha='right', labelpad=40)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=4)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=HOUSE_COLORS.get(h, 'grey'), markersize=7, label=h)
        for h in houses
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               title='House', title_fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pair_plot.py <dataset.csv>')
        sys.exit(1)
    pair_plot(sys.argv[1])
