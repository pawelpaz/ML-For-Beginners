# Lecture: Building Good Training Datasets — Data Preprocessing

In this lesson, we are diving into what is arguably the most critical stage of any machine learning pipeline: **Data Preprocessing**. While building fancy models is exciting, the quality of your data and the amount of useful information it contains determine how well your algorithm can actually learn. We must examine and preprocess our datasets before feeding them to any algorithm.

![data](./images/data.jpg)

---

## 1. Dealing with Missing Data

Real-world data is often missing values due to collection errors, non-applicable measurements, or blank survey fields. These appear as blank spaces, `NaN` (Not a Number), or `NULL` indicators. 

### Identifying Missing Values
Using `pandas`, we can scan for these gaps efficiently. For large datasets, manual inspection is tedious, so we use the `isnull()` method to find missing cells and `sum()` to count them per column.

```python
import pandas as pd
import numpy as np

# Sample dataset with missing values
data = {
    'Feature_A': [1.0, 2.0, np.nan, 4.0, 5.0],
    'Feature_B': [5.0, np.nan, np.nan, 8.0, 10.0],
    'Category': ['Red', 'Blue', 'Red', np.nan, 'Green']
}
df = pd.DataFrame(data)

# Count missing values per column
print(df.isnull().sum())
```

### Strategies for Handling Gaps
1.  **Elimination:** You can drop rows (`axis=0`) or columns (`axis=1`) containing missing data using the `dropna` method. 
    * **Pros:** Quick and easy.
    * **Cons:** You risk losing too much data, which can make reliable analysis impossible or hinder a classifier's ability to discriminate between classes.
2.  **Imputation:** If removal isn't feasible, we use **interpolation techniques**. 
    * **Mean Imputation:** Replacing a missing value with the mean of the entire feature column.
    * **Median/Most Frequent:** Alternatives for the `strategy` parameter. "Most frequent" is particularly useful for categorical data like color names.

```python
# 1. Elimination: Drop rows with any NaN values
df_dropped = df.dropna(axis=0)

# 2. Basic Imputation: Fill missing values with the mean using Pandas
df['Feature_A_filled'] = df['Feature_A'].fillna(df['Feature_A'].mean())
```

---

## 2. The Scikit-Learn Estimator API

To perform transformations like imputation, scikit-learn provides a consistent **Transformer API**. 
* **`fit`:** Used to learn parameters (like the mean) from the training data.
* **`transform`:** Uses those learned parameters to actually change the data.

```python
from sklearn.impute import SimpleImputer

# Initialize the imputer for mean strategy
imputer = SimpleImputer(strategy='mean')

# We'll use only numerical columns for this
numerical_cols = ['Feature_A', 'Feature_B']

# Fit and transform the data
# Note: imputer returns a NumPy array, so we wrap it back in a DataFrame or assign to columns
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

print(df)from 
```

> **Note:** We always `fit` on the training data *only*, then use that fitted instance to `transform` both the training and the test sets to ensure consistency.

---

## 3. Handling Categorical Data

We must distinguish between two types of categorical features:
* **Ordinal Features:** Values that can be sorted or ordered (e.g., T-shirt size: XL > L > M).
* **Nominal Features:** Values that do not imply any order (e.g., T-shirt color: red is not "larger" than blue).

### Encoding Strategies
* **Ordinal Mapping:** We manually define a mapping so the algorithm interprets the order correctly.

```python
# Sample Ordinal Data
df_size = pd.DataFrame({'Size': ['L', 'M', 'XL', 'M', 'S']})

# Define the mapping (e.g., {'XL': 4, 'L': 3, 'M': 2, 'S': 1})
size_mapping = {'XL': 4, 'L': 3, 'M': 2, 'S': 1}

# Apply mapping
df_size['Size'] = df_size['Size'].map(size_mapping)
print(df_size)
```

* **One-Hot Encoding:** A common mistake is encoding nominal features as ordered integers (0, 1, 2), which leads models to assume a mathematical hierarchy that doesn't exist. Instead, we use **one-hot encoding** to create a "dummy" feature for each unique category.
    * **Multi-collinearity:** To avoid numerically unstable estimates, it is best practice to remove one redundant column from the encoded array.

```python
# One-hot encoding with pandas
# drop_first=True avoids the "dummy variable trap" (multi-collinearity)
df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=True)
print(df_encoded.head())
```

---

## 4. Partitioning the Dataset

To evaluate how well our model generalizes, we split data into a **training set** and a **test set**.

* **Common Splits:** 60:40, 70:30, or 80:20 are standard, though 90:10 is fine for very large datasets.
* **Stratification:** Using `stratify=y` ensures both sets have the same class proportions as the original dataset.

```python
from sklearn.model_selection import train_test_split

# X = Features, y = Target/Label
X = df_encoded.drop('Feature_A', axis=1) 
y = df_encoded['Feature_A']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=None 
)

print(f"Training set rows: {X_train.shape[0]} | Test set rows: {X_test.shape[0]}")
```

---

## 5. Feature Scaling

Most algorithms (except scale-invariant ones like decision trees) perform better when features are on the same scale.

### Normalization vs. Standardization

| Technique | Description | Formula |
| :--- | :--- | :--- |
| **Normalization** | Rescales data to a range of [0, 1] (min-max scaling). | x_norm = (x - x_min) / (x_max - x_min) |
| **Standardization** | Centers features at mean 0 with standard deviation 1. | x_std = (x - mu) / sigma |

**Why Standardization?** It is often more practical for optimization algorithms like gradient descent because it centers the data, making it easier to learn weights while maintaining information about outliers.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalization (Min-Max Scaling)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardization (Z-score Scaling)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```

---

## 6. Handling Outliers

**Outliers** are data points that differ significantly from the rest of the observations. They can be caused by measurement errors (e.g., a faulty sensor) or represent genuine but rare anomalies.

### Why Outliers Matter
Many machine learning algorithms—especially those based on distance or central tendency like Linear Regression, K-Means Clustering, and Logistic Regression—are highly sensitive to extreme values. A single "garbage" data point can drastically skew your model's coefficients or cluster centers.

### The IQR (Interquartile Range) Method
The most common statistical method for identifying outliers is the IQR method. It focuses on the "middle 50%" of the data.

1.  **Calculate Q1** (25th percentile) and **Q3** (75th percentile).
2.  **Compute IQR = Q3 - Q1.**
3.  **Define Bounds:**
    * **Lower Bound:** Q1 - 1.5 * IQR
    * **Upper Bound:** Q3 + 1.5 * IQR

Any value falling outside these bounds is statistically considered an outlier.

### Code Example: Detection and Filtering

```python
import pandas as pd
import numpy as np

# Sample dataset with a clear outlier (999)
df_out = pd.DataFrame({'Scores': [10, 12, 12, 13, 15, 11, 14, 18, 12, 999]})

# 1. Calculate Quartiles
Q1 = df_out['Scores'].quantile(0.25)
Q3 = df_out['Scores'].quantile(0.75)
IQR = Q3 - Q1

# 2. Define the bounds
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# 3. Identify and Filter
outliers = df_out[(df_out['Scores'] < lower_limit) | (df_out['Scores'] > upper_limit)]
df_cleaned = df_out[(df_out['Scores'] >= lower_limit) & (df_out['Scores'] <= upper_limit)]

print(f"Detected Outliers:\n{outliers}")
print(f"Dataset size after removal: {len(df_cleaned)}")
```

### Alternative Strategies
* **Z-Score:** If your data is normally distributed, you can flag points that are more than 3 standard deviations away from the mean.
* **Winsorization (Clipping):** Instead of deleting outliers, you "cap" them. For example, any value above the 95th percentile is set exactly to the 95th percentile value to limit its influence without losing the data point entirely.