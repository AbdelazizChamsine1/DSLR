# Explanation — Math & Logic Behind DSLR

This document walks through every concept used in the project, from data preprocessing to the final prediction.

---

## 1. The Problem

We have a dataset of Hogwarts students, each belonging to one of four houses:
`Gryffindor`, `Hufflepuff`, `Ravenclaw`, `Slytherin`.

Given a student's course scores, we want to **predict their house**.

This is a **multi-class classification** problem. We solve it using **logistic regression** with a **one-vs-all** strategy.

---

## 2. Data Preprocessing

### 2.1 Parsing

The CSV is parsed manually — no pandas. Each column is stored as a list of strings, then converted to floats where applicable.

### 2.2 Handling Missing Values

Some students are missing scores for certain courses. During training, missing values are **imputed with the column mean**. This avoids discarding entire rows and keeps the dataset size intact.

### 2.3 Feature Normalization (Z-score)

Raw scores vary wildly in scale (e.g., Astronomy: -600 to 0, Charms: -250 to -200). If we feed these raw values into gradient descent, features with large scales will dominate the gradient updates.

We apply **z-score normalization** to each feature:

$$x' = \frac{x - \mu}{\sigma}$$

Where:
- $\mu$ is the mean of the feature across all training samples
- $\sigma$ is the standard deviation

After normalization, every feature has **mean ≈ 0** and **std ≈ 1**. This makes gradient descent converge faster and more stably.

> The means and stds computed on the training set are saved in `weights.json` and reused at prediction time — we never normalize using test set statistics.

---

## 3. Logistic Regression

### 3.1 Why Not Linear Regression?

Linear regression predicts a continuous value. For classification, we need probabilities bounded between 0 and 1. We apply the **sigmoid function** to a linear combination of features to achieve this.

### 3.2 The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Output is always in **(0, 1)** — interpretable as a probability
- $\sigma(0) = 0.5$
- As $z \to +\infty$, $\sigma(z) \to 1$
- As $z \to -\infty$, $\sigma(z) \to 0$

### 3.3 The Hypothesis

For a sample $x$ with feature vector $[x_1, x_2, \ldots, x_n]$, the model predicts:

$$h_\theta(x) = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n)$$

Where:
- $\theta_0$ is the **bias** (intercept)
- $\theta_1 \ldots \theta_n$ are the **weights** for each feature
- $h_\theta(x) \in (0, 1)$ is the predicted probability of belonging to the positive class

---

## 4. The Cost Function (Log Loss)

We cannot use mean squared error for logistic regression — it produces a non-convex surface with many local minima. Instead we use **binary cross-entropy** (log loss):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

Where:
- $m$ is the number of training samples
- $y^{(i)} \in \{0, 1\}$ is the true label for sample $i$
- $h_\theta(x^{(i)})$ is the predicted probability for sample $i$

**Intuition:**
- When $y = 1$: the cost is $-\log(h)$. If $h \to 1$ (correct), cost $\to 0$. If $h \to 0$ (wrong), cost $\to \infty$.
- When $y = 0$: the cost is $-\log(1 - h)$. Same logic in reverse.

This function is **convex** — it has a single global minimum that gradient descent can reliably find.

---

## 5. Gradient Descent

We minimize $J(\theta)$ by iteratively updating the weights in the direction of steepest descent.

### 5.1 The Gradient

The partial derivative of $J$ with respect to weight $\theta_j$ is:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

For the bias term ($j = 0$, where $x_0 = 1$):

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)$$

### 5.2 The Update Rule

$$\theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}$$

Where $\alpha$ is the **learning rate** — a small positive number that controls the step size.

- Too large: overshoots the minimum, diverges
- Too small: converges correctly but very slowly
- We use $\alpha = 0.1$ as default

This is **batch gradient descent** — we compute the gradient over the entire training set at each step, which is stable but slower than stochastic variants.

---

## 6. One-vs-All (OvA) Strategy

Logistic regression is inherently **binary** (two classes). To handle four houses, we train **4 separate classifiers**:

| Classifier | Positive class ($y=1$) | Negative class ($y=0$) |
|------------|------------------------|------------------------|
| Classifier 1 | Gryffindor | Hufflepuff + Ravenclaw + Slytherin |
| Classifier 2 | Hufflepuff | Gryffindor + Ravenclaw + Slytherin |
| Classifier 3 | Ravenclaw  | Gryffindor + Hufflepuff + Slytherin |
| Classifier 4 | Slytherin  | Gryffindor + Hufflepuff + Ravenclaw |

Each classifier learns to answer: *"Is this student in my house, or not?"*

### Prediction

At inference time, we run all 4 classifiers on the input and pick the house with the **highest probability**:

$$\hat{y} = \arg\max_{k \in \{1,2,3,4\}} h_{\theta_k}(x)$$

---

## 7. Feature Selection

Not all courses carry useful information for separating houses. Using the **pair plot**, we look for features where house clusters are visually distinct.

**Excluded features:**
- `Arithmancy` — scores are nearly uniform across all houses (no discriminative power)
- `Potions` — high overlap between houses
- `Care of Magical Creatures` — high overlap between houses

**Kept features (10 total):**
Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Charms, Flying.

---

## 8. Descriptive Statistics (from scratch)

All statistics used in `describe.py` are implemented without any library functions:

| Statistic | Formula |
|-----------|---------|
| Count     | Iterate and increment counter |
| Mean      | $\bar{x} = \frac{1}{n}\sum x_i$ |
| Std       | $s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$ (sample std) |
| Min / Max | Linear scan |
| Percentile (Q1, Q2, Q3) | Sort, interpolate between neighbors at index $\frac{p}{100}(n-1)$ |

---

## 9. Pearson Correlation (for scatter plot)

To find the two most similar features, we compute the **Pearson correlation coefficient**:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2} \cdot \sqrt{\sum(y_i-\bar{y})^2}}$$

- $r \in [-1, 1]$
- $|r| \approx 1$ means strong linear correlation
- $|r| \approx 0$ means no linear relationship

We pick the pair with the highest $|r|$ — typically **Astronomy** and **Defense Against the Dark Arts**.

---

## 10. Summary Flow

```
dataset_train.csv
       │
       ▼
  Parse CSV  ──►  Filter NaN  ──►  Compute mean/std per feature
       │
       ▼
  Z-score normalize each feature
       │
       ▼
  For each house (OvA):
    - Label: 1 if house matches, else 0
    - Run gradient descent for N epochs
    - Save weights θ
       │
       ▼
  weights.json  (weights + means + stds)
       │
       ▼
dataset_test.csv  ──►  Normalize (same mean/std)  ──►  Run 4 classifiers
       │
       ▼
  argmax(probabilities)  ──►  houses.csv
```
