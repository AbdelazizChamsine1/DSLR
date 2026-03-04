import math

from utils.data_utils import get_numerical_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

HOUSE_COLORS = {
    'Gryffindor': '#c0392b',
    'Hufflepuff': '#f1c40f',
    'Ravenclaw':  '#2980b9',
    'Slytherin':  '#27ae60',
}

NON_COURSE_COLUMNS = {'Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

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


def safe_float(val):
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def group_by_house(dataset: dict, course: str) -> dict:
    """Return {house: [scores]} for a given course."""
    house_col = dataset.get('Hogwarts House', [])
    course_col = dataset.get(course, [])
    groups = {h: [] for h in HOUSES}
    for house, val in zip(house_col, course_col):
        if val in ('', 'NA', 'NaN'):
            continue
        try:
            score = float(val)
            if not math.isnan(score) and house in groups:
                groups[house].append(score)
        except ValueError:
            continue
    return groups


def get_house_data(dataset: dict, courses: list, houses: list) -> dict:
    """Return {house: {course: [values]}} with NaNs dropped per column."""
    house_col = dataset.get('Hogwarts House', [])
    n_rows = len(house_col)
    result = {h: {c: [] for c in courses} for h in houses}

    for row_idx in range(n_rows):
        house = house_col[row_idx].strip() if row_idx < len(house_col) else ''
        if house not in result:
            continue
        for course in courses:
            col = dataset.get(course, [])
            val = safe_float(col[row_idx]) if row_idx < len(col) else None
            if val is not None:
                result[house][course].append(val)

    return result


def get_paired_values(col_a: list, col_b: list) -> tuple:
    """Return two aligned float lists, skipping rows where either value is missing."""
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


# ---------------------------------------------------------------------------
# Statistical helpers for plots
# ---------------------------------------------------------------------------

def pearson_correlation(x: list, y: list) -> float:
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


def homogeneity_score(groups: dict) -> float:
    """Lower = more homogeneous across houses."""
    means = []
    stds  = []
    for data in groups.values():
        if data:
            m = sum(data) / len(data)
            means.append(m)
            s = math.sqrt(sum((x - m) ** 2 for x in data) / len(data))
            stds.append(s)
    if not means:
        return float('inf')
    mean_of_means = sum(means) / len(means)
    spread = math.sqrt(sum((m - mean_of_means) ** 2 for m in means) / len(means))
    avg_std = sum(stds) / len(stds) if stds else 1
    return spread / avg_std if avg_std != 0 else float('inf')
