
"""Functions for plotting the Titanic data."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from titanic.config import COLOR_LIST


def plot_count_pairs(
    df: pd.DataFrame,
    feature: str,
    hue: str = "set",
    color_list: list[str] = COLOR_LIST,
    title: str | None = None,
) -> plt.Axes:
    """Plot the number of passengers in each group."""
    n_levels = df[hue].nunique()
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=df, hue=hue, palette=color_list[:n_levels], ax=ax)
    ax.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(title or f"Number of passengers / {feature}")
    plt.show()
    return ax


def plot_distribution_pairs(
    df: pd.DataFrame,
    feature: str,
    hue: str = "set",
    color_list: list[str] = COLOR_LIST,
    title: str | None = None,
) -> plt.Axes:
    """Plot the distribution of a feature for different groups."""
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, level in enumerate(df[hue].dropna().unique()):
        sns.histplot(
            df.loc[df[hue] == level, feature].dropna(),
            color=color_list[i % len(color_list)],
            ax=ax,
            label=str(level),
        )
    ax.legend(title=hue)
    ax.set_title(title or f"Distribution of {feature}")
    plt.show()
    return ax
