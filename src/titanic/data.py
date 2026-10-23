
"""Functions for loading and checking the Titanic data."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)


def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the training and test data and label which set each row belongs to."""
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df["set"] = "train"
    all_df.loc[all_df["Survived"].isna(), "set"] = "test"
    return all_df


def missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check the number and percentage of missing values in each column."""
    total = df.isnull().sum()
    percent = total / len(df) * 100
    summary = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    summary["Types"] = [str(df[col].dtype) for col in df.columns]
    return summary.T


def most_frequent(df: pd.DataFrame) -> pd.DataFrame:
    """Find the most common value in each column and how often it appears."""
    total = df.count()
    summary = pd.DataFrame({"Total": total})
    items, vals = [], []
    for col in df.columns:
        counts = df[col].value_counts()
        if counts.empty:
            items.append(None)
            vals.append(0)
        else:
            items.append(counts.index[0])
            vals.append(counts.iloc[0])
    summary["Most frequent item"] = items
    summary["Frequence"] = vals
    summary["Percent from total"] = (summary["Frequence"] / total * 100).round(3)
    return summary.T


def unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many different values each column has."""
    summary = pd.DataFrame({"Total": df.count(), "Uniques": df.nunique()})
    return summary.T
