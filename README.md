# DSLR — Data Science x Logistic Regression
### Harry Potter and the Data Scientist

Recreate the Hogwarts Sorting Hat using a logistic regression classifier trained from scratch on student data. The model assigns each student to one of the four houses: **Gryffindor**, **Hufflepuff**, **Ravenclaw**, or **Slytherin**.

---

## Project Structure

```
DSLR/
├── data/
│   ├── dataset_train.csv       # Labeled training data
│   └── dataset_test.csv        # Unlabeled test data
├── src/
│   ├── describe.py             # Descriptive statistics
│   ├── histogram.py            # Score distribution visualization
│   ├── scatter.py              # Most correlated features
│   ├── pair_plot.py            # Full feature scatter matrix
│   ├── logreg_train.py         # Train the model
│   ├── logreg_predict.py       # Generate predictions
│   └── evaluate.py             # Measure accuracy on a validation split
├── utils/
│   ├── data_utils.py           # CSV parsing, NaN handling, normalization
│   ├── stat_utils.py           # Stats from scratch (mean, std, percentile...)
│   └── plot_utils.py           # Shared plot helpers and constants
├── weights.json                # Saved model weights (generated after training)
├── houses.csv                  # Output predictions (generated after predict)
├── README.md
└── explanation.md
```

---

## Requirements

- Python 3.10+
- `matplotlib` (only for visualization scripts)

```bash
pip install matplotlib
```

> All statistical computations (mean, std, percentile, etc.) are implemented from scratch — no NumPy, no pandas, no sklearn.

---

## Usage

### 1. Explore the data

```bash
# Descriptive statistics for all numerical features
python src/describe.py data/dataset_train.csv

# Which course has the most homogeneous score distribution across houses?
python src/histogram.py data/dataset_train.csv

# Which two features are the most correlated?
python src/scatter.py data/dataset_train.csv

# Full scatter matrix for feature selection
python src/pair_plot.py data/dataset_train.csv
```

### 2. Train the model

```bash
python src/logreg_train.py data/dataset_train.csv weights.json
```

Optional arguments:
```bash
python src/logreg_train.py data/dataset_train.csv weights.json [lr] [epochs]
# Example: lr=0.1, epochs=1000 (defaults)
```

### 3. Predict

```bash
python src/logreg_predict.py data/dataset_test.csv weights.json houses.csv
```

Output format (`houses.csv`):
```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
...
```

### 4. Evaluate accuracy

```bash
python src/evaluate.py data/dataset_train.csv
```

Optional arguments:
```bash
python src/evaluate.py data/dataset_train.csv [val_ratio] [lr] [epochs]
# Example: 0.2 0.1 1000 (defaults)
```

Sample output:
```
Overall accuracy: 318/320  =  99.37%

House         Correct   Total     Acc
----------------------------------------
Gryffindor        82      83   98.80%
Hufflepuff        79      80   98.75%
Ravenclaw         85      85  100.00%
Slytherin         72      72  100.00%
```

---

## Features Used for Classification

After pair plot analysis, 10 features were selected (Arithmancy, Potions, and Care of Magical Creatures were excluded due to low discriminative power):

| Feature |
|---------|
| Astronomy |
| Herbology |
| Defense Against the Dark Arts |
| Divination |
| Muggle Studies |
| Ancient Runes |
| History of Magic |
| Transfiguration |
| Charms |
| Flying |

---

## Target Accuracy

≥ **98%** on the test set, evaluated using scikit-learn's `accuracy_score`.

---

## See Also

- [`explanation.md`](explanation.md) — detailed walkthrough of the math and logic behind the model.
