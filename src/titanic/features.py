
"""Functions for creating new features from the Titanic data."""

import numpy as np
import pandas as pd

AGE_BINS = [16, 32, 48, 64]
FARE_BINS = [7.91, 14.454, 31]
TITLE_MAP = {
    "Mlle.": "Miss.",
    "Ms.": "Miss.",
    "Mme.": "Mrs.",
    **{
        t: "Rare"
        for t in [
            "Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.",
            "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.",
        ]
    },
}


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate family size for each passenger."""
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def _interval(values: pd.Series, upper_bounds: list[float]) -> np.ndarray:
    """Put values into groups based on the given boundaries."""
    conditions = [values <= upper_bounds[0]]
    conditions += [
        (values > lo) & (values <= hi) for lo, hi in zip(upper_bounds, upper_bounds[1:])
    ]
    conditions.append(values > upper_bounds[-1])
    return np.select(conditions, range(len(conditions)), default=0).astype(float)


def add_age_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Divide passengers into five age groups."""
    df = df.copy()
    df["Age Interval"] = _interval(df["Age"], AGE_BINS)
    return df


def add_fare_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Divide ticket fares into four groups."""
    df = df.copy()
    df["Fare Interval"] = _interval(df["Fare"], FARE_BINS)
    return df


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new column combining sex and passenger class."""
    df = df.copy()
    df["Sex_Pclass"] = df["Sex"].str[0].str.upper() + "_C" + df["Pclass"].astype(str)
    return df


def parse_names(name: str) -> tuple[str, str, str, str | None]:
    """Separate a passenger's name into family name, title, given name and maiden name."""
    family_name, rest = name.split(",", 1)
    title, given = rest.split(".", 1)
    title = title.strip() + "."
    if "(" in given:
        given_name, maiden = given.split("(", 1)
        return family_name, title, given_name, maiden.rstrip(")")
    return family_name, title, given, None


def add_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Add separate columns for different parts of passenger names."""
    df = df.copy()
    parts = df["Name"].apply(parse_names)
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = pd.DataFrame(
        parts.tolist(), index=df.index
    )
    return df


def add_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """Group passengers into single, small or large families."""
    df = df.copy()
    size = df["Family Size"]
    df["Family Type"] = np.select(
        [size == 1, size < 5], ["Single", "Small"], default="Large"
    )
    return df


def add_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Group passenger titles into common titles and rare titles."""
    df = df.copy()
    df["Titles"] = df["Title"].replace(TITLE_MAP)
    return df


def survival_rate_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calculate the average survival rate for each group."""
    return df[cols + ["Survived"]].groupby(cols, as_index=False).mean()


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Change sex into numbers, with female as 1 and male as 0."""
    df = df.copy()
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all the functions above to create new features."""
    df = add_family_size(df)
    df = add_age_interval(df)
    df = add_fare_interval(df)
    df = add_sex_pclass(df)
    df = add_name_parts(df)
    df = add_family_type(df)
    df = add_titles(df)
    return df
