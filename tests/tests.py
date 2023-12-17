import sys
import pytest
import pandas as pd

sys.path.append('src')

from fselect import compute_entropy, get_correlated_columns, rank_features


def test_compute_entropy():
    # Test a simple dataframe
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [4, 5, 6, 7]
    })
    entropy = compute_entropy(df)
    assert entropy > 5.0

    # Test a dataframe with correlated columns
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [4, 5, 6, 7],
        "c": [0, 1, 2, 3]
    })
    entropy = compute_entropy(df)
    assert entropy > 5.0


def test_get_correlated_columns():
    # Test a simple dataframe
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [9, 5, 6, 7]
    })
    correlated_columns = get_correlated_columns(df, 0.999)
    assert correlated_columns == {'a': ['a'], 'b': ['b']}

    # Test a dataframe with correlated columns
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [9, 5, 6, 7],
        "c": [0, 1, 2, 3]
    })
    correlated_columns = get_correlated_columns(df, 0.999)
    assert correlated_columns == {'a': ['a', 'c'], 'b': ['b'], 'c': ['a', 'c']}


def test_rank_features():
    # Test a simple dataframe
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [9, 5, 6, 7]
    })
    rankings = rank_features(df)
    assert rankings.shape == (2, 2)
    assert rankings.iloc[0]['feature'] == "b"
    assert len(rankings["feature"].values) == len(["a", "b"])
    assert all([a == b for a, b in zip(rankings["feature"].values, ["b", "a"])])

    # Test a dataframe with correlated columns
    df = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "b": [9, 5, 6, 7],
        "c": [0, 1, 2, 3]
    })
    rankings = rank_features(df)
    assert rankings.shape == (3, 2)
    assert rankings.iloc[0]['feature'] == "b"


if __name__ == "__main__":
    test_compute_entropy()
    test_get_correlated_columns()
    test_rank_features()
    print("All tests passed!")
