import math

def parse_csv(filepath: str) -> dict:
    dataset = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    headers = lines[0].strip().split(',')
    for header in headers:
        dataset[header] = []
    
    for line in lines[1:]:
        values = line.strip().split(',')
        for i, value in enumerate(values):
            dataset[headers[i]].append(value.strip())
    
    return dataset


def filter_nan(data: list) -> list[float]:
    cleaned = []
    for value in data:
        if value == '' or value == 'NA' or value == 'NaN':
            continue
        try:
            f = float(value)
            if not math.isnan(f):
                cleaned.append(f)
        except ValueError:
            continue
    return cleaned


def get_numerical_columns(dataset: dict) -> dict:
    numerical = {}
    for column, values in dataset.items():
        cleaned = filter_nan(values)
        if len(cleaned) > 0:
            try:
                float(cleaned[0])
                numerical[column] = cleaned
            except (ValueError, IndexError):
                continue
    return numerical


def normalize(data: list[float], mean: float, std: float) -> list[float]:
    if std == 0:
        return [0.0 for _ in data]
    normalized = []
    for value in data:
        normalized.append((value - mean) / std)
    return normalized