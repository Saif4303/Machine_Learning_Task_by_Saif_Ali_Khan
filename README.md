# Nested Time Series Cross-Validation

This project implements a nested time series cross-validation strategy for grouped forecasting using the scikit-learn format. The nested cross-validation is tailored to handle time series data with a hierarchical or grouped structure, ensuring proper time-based splits without data leakage.

## Objectives

1. **Implement Nested Time Series Cross-Validation Strategy**:
   - User provides the dataset, time column, and the number of folds to generate.
   - Use **"day"** as a single time unit for splitting the data.

2. **Adherence to scikit-learn Format**:
   - The class works on pandas dataframes and a datetime column name.
   - The implementation is inspired by the [KFold CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).

3. **Model Evaluation**:
   - Build a time series model and evaluate it using the custom cross-validation method.

## Files

- `cv.py`: Contains the `NestedCV` class and unit tests.
- `NestedCV.ipynb`: Script to fit a time series model (ARIMA) and evaluate it using the nested cross-validation method.

## Prerequisites

- Python 3.11.4  or above
- Pandas
- Scikit-learn
- Statsmodels

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Saif4303/Machine_Learning_Task_by_Saif_Ali_Khan.git
   ```

2. Install the required packages:
   ```
   pip install pandas
   pip install scikit-learn
   pip install statsmodels
   ```

## Usage

### NestedCV Class

The `NestedCV` class is designed to split the dataset into nested time series cross-validation folds.

#### Example

```python
import pandas as pd

from types import GeneratorType

class NestedCV:
    def __init__(self, k):
        self.k = k

    def split(self, data, date_column):
        # Write Nested Cross validation for time series
        # logic here and return the train and validate dataframes
        # return a generator object for the splits

if __name__ == "__main__":
    # load dataset
    data = pd.read_csv("paht to dataset")
    data["date"] = pd.to_datetime(data["date"])

    # nested cv
    k = 3
    cv = NestedCV(k)
    splits = cv.split(data, "date")

    # check return type
    assert isinstance(splits, GeneratorType)

    # check return types, shapes, and data leaks
    count = 0
    for train, validate in splits:
        
        # types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(validate, pd.DataFrame)

        # shape
        assert train.shape[1] == validate.shape[1]

        # data leak
        assert train["date"].max() <= validate["date"].min()

        count += 1

    # check number of splits returned
    assert count == k
```

### Model Evaluation

A script to fit a ARIMA model and evaluate it using the nested cross-validation method.

#### Example

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Defining a function to fit ARIMA model and evaluate
def fit_arima(train, validate):
    # Fit ARIMA model
    model = ARIMA(train['quantity'], order=(2, 0, 1))
    model_fit = model.fit()

    # Making predictions
    forecast = model_fit.forecast(steps=len(validate))

    # Calculating error (e.g., RMSE)
    rmse = mean_squared_error(validate['quantity'], forecast, squared=False)

    return rmse

# Initializing lists to store evaluation results
rmse_scores = []

# Performing nested cross-validation
cv = NestedCV(k)
splits = cv.split(data, "date")

for train, validate, test in splits:
    rmse = fit_arima(train, validate)
    rmse_scores.append(rmse)

# Calculating mean RMSE across all folds
mean_rmse = np.mean(rmse_scores)
print("Mean RMSE:", mean_rmse)


# Sample data
data = pd.DataFrame({
    'date': pd.date_range(start='2022-01-01', periods=50),
    'city': ['A']*25 + ['B']*25,
    'quantity': range(50)
})

# Initialize nested CV
k = 5
cv = NestedCV(k)
splits = cv.split(data, "date")

# Initialize lists to store evaluation results
rmse_scores = []

# Perform nested cross-validation
for train, validate, test in splits:
    rmse = fit_sarima(train, validate)
    rmse_scores.append(rmse)

# Calculate mean RMSE across all folds
mean_rmse = sum(rmse_scores) / len(rmse_scores)
print("Mean RMSE:", mean_rmse)
```


## Acknowledgments

- The nested cross-validation approach is inspired by the need for proper evaluation of time series models with hierarchical/grouped data.
- The implementation follows the guidelines and format of scikit-learn's cross-validation utilities.

---
