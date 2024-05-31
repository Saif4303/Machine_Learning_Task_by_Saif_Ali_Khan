import pandas as pd
import numpy as np
from types import GeneratorType
from sklearn.model_selection import TimeSeriesSplit

class NestedCV:
    def __init__(self, k):
        self.k = k

    def split(self, data, date_column):
        # Sort the data by date
        data = data.sort_values(by=date_column)

        # Initialize the outer time series split
        outer_tscv = TimeSeriesSplit(n_splits=self.k + 1)

        for train_val_index, test_index in outer_tscv.split(data):
            # Get the train_val split and test split
            train_val, test = data.iloc[train_val_index], data.iloc[test_index]

            # Initialize the inner time series split
            inner_tscv = TimeSeriesSplit(n_splits=self.k)

            for train_index, val_index in inner_tscv.split(train_val):
                # Get the train and validation split
                train, val = train_val.iloc[train_index], train_val.iloc[val_index]

                yield train, val, test


if __name__ == "__main__":
    # Loading and preprocessing dataset
    data = pd.read_csv(r"C:\Users\ASIF ALI KHAN\Dropbox\PC\Downloads")
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by="date")
    data = data.fillna(method='ffill')
    data = data.drop_duplicates()
    data = data.drop(['id', 'lat', 'long', 'pop'], axis=1)  # dropping unneccesary columns

    # Aggregating data for duplicate dates
    data = data.groupby('date').agg({
        'shop': 'first',
        'brand': 'first',
        'container': 'first',
        'capacity': 'first',
        'price': 'mean',
        'quantity': 'sum'
    }).reset_index()

    # Resampling the data to a daily frequency
    data = data.set_index("date").resample('D').asfreq().reset_index()
    data = data.fillna(0)

    # Initializing NestedCV with k-folds
    k = 5
    cv = NestedCV(k)
    splits = cv.split(data, "date")

    # Checking return type
    assert isinstance(splits, GeneratorType)

    # Checking return types, shapes, and data leaks
    count = 0
    for train, validate, test in splits:
        # Types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(validate, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

        # Shape
        assert train.shape[1] == validate.shape[1] == test.shape[1]

        # Data leak
        assert train["date"].max() <= validate["date"].min()
        assert validate["date"].max() <= test["date"].min()

        count += 1

    # Checking number of splits returned
    assert count == k * (k + 1)
    print(f"Number of splits: {int(count / (k + 1))}")
