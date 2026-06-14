"""A-RANK feature ranking for clustering.

Implements a modified version of the A-RANK algorithm from
"Dash, M. and Liu, H. — Feature Selection for Clustering". It ranks the
continuous features of a pandas DataFrame by an entropy measure, so the most
informative features for downstream clustering can be selected.
"""

from copy import deepcopy
import math

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def rank_features(
        dataframe: pd.DataFrame,
        remove_correlated_columns: bool = False,
        correlation_threshold: float = 0.999
) -> pd.DataFrame:
    """Rank features by an entropy measure for clustering (A-RANK).

    For each feature, the feature (or its correlated group) is dropped and the
    entropy of the remaining columns is computed; features whose removal leaves
    higher entropy are ranked as more important.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input data with continuous, normalized columns to be ranked.
    remove_correlated_columns : bool
        If True, drop each feature's highly-correlated group (rather than just
        the feature itself) when measuring entropy, since correlated columns
        skew the measure.
    correlation_threshold : float
        Absolute-correlation threshold defining "closely related" columns when
        ``remove_correlated_columns`` is True. Defaults to 0.999.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns ``feature``, ``entropy`` and ``rank``,
        sorted by entropy descending (rank 1 = most important).
    """
    entropy_values = []
    if remove_correlated_columns:
        correlated_columns = get_correlated_columns(
            dataframe, correlation_threshold
        )

    for feature in dataframe.columns:
        features_to_drop = []
        if remove_correlated_columns:
            features_to_drop += correlated_columns[feature]
        else:
            features_to_drop.append(feature)
        dataframe_dropped_features = dataframe.drop(columns=features_to_drop)

        if len(dataframe_dropped_features.columns) < 1:
            raise Exception("Empty Dataframe! \nDataframe might have only one \
                            feature or only one non-correlated feature if \
                            remove_correlated_columns is True")
        entropy = compute_entropy(dataframe_dropped_features)
        entropy_values.append(entropy)

    feature_entropies = pd.DataFrame({
        "feature": dataframe.columns,
        "entropy": entropy_values
    })

    rankings = feature_entropies.sort_values(
        by="entropy", ascending=False).reset_index(drop=True)
    rankings["rank"] = rankings.index + 1
    return rankings


def compute_entropy(dataframe: pd.DataFrame) -> float:
    """Compute the total entropy of a dataframe as defined in the A-RANK paper.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe from rank_features.

    Returns
    -------
    total_entropy: float
        The calculated total entropy of the dataframe based on the
        similarity-matrix formulation suggested in the paper.
    """

    dataframe = deepcopy(dataframe)
    dataframe = dataframe.reset_index(drop=True)
    df_pairwise_distances = pairwise_distances(dataframe.to_numpy())
    alpha = -math.log(0.5) / np.matrix.mean(np.asmatrix(df_pairwise_distances))
    df_similarity_matrix = np.exp(-alpha * df_pairwise_distances)
    df_entropies = - ((df_similarity_matrix * np.log2(df_similarity_matrix))
                      + ((1 - df_similarity_matrix)
                      * np.log2(1 - df_similarity_matrix)))
    total_entropy = np.nansum(df_entropies) / 2.0

    return total_entropy


def get_correlated_columns(dataframe: pd.DataFrame,
                           correlation_threshold: float
                           ) -> dict:
    """Map each column to the columns correlated with it above a threshold.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe from rank_features.
    correlation_threshold: float
        The threshold value to identify correlated columns.

    Returns
    -------
    correlated_columns: dict
        Dictionary with each column as a key and a list of correlated
        columns (always including the column itself) as the value.
    """

    correlation_matrix = dataframe.corr()
    correlated_columns = {}
    for col in correlation_matrix.columns:
        indices = correlation_matrix.index[
            abs(correlation_matrix[col]) > correlation_threshold
        ].tolist()
        correlated_columns[col] = indices
    return correlated_columns
