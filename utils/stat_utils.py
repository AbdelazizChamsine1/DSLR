import math

def get_count(data: list[float]) -> int:
    count = 0
    for _ in data:
        count += 1
    return count


def get_sum(data: list[float]) -> float:
    total = 0.0
    for n in data:
        total += n
    return total


def get_mean(data: list[float]) -> float:
    l = get_count(data)
    if l == 0:
        return 0.0
    return get_sum(data) / l


def get_min(data: list[float]) -> float:
    minimum = data[0]
    for n in data:
        if n < minimum:
            minimum = n
    return minimum


def get_max(data: list[float]) -> float:
    maximum = data[0]
    for n in data:
        if n > maximum:
            maximum = n
    return maximum


def get_std(data: list[float]) -> float:
    l = get_count(data)
    if l == 0:
        return 0.0
    mean = get_mean(data)
    total = 0.0
    for n in data:
        total += (n - mean) ** 2
    return math.sqrt(total / l)


def get_percentile(data: list[float], p: float) -> float:
    sorted_data = sorted(data)
    l = get_count(sorted_data)
    index = (p / 100) * (l - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= l:
        return sorted_data[lower]
    fraction = index - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])