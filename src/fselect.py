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
    """
    Main function calling other functions to calculate entropy
    and ranking the features.
    Implements a modified version of ARANK algorithm as defined in the paper
    "Dash, M. and Lie, H. Feature Selection for Clustering"

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe with continuos (normalized)
        data with columns which are to be ranked on
        the basis of importance for further clustering.

    remove_correlated_columns : bool
        Optional parameter to remove any
        closely related columns since it effects
        the entropy measure and hence the rankings.
        More details on github readme.

    correlation_threshold : float
        If above parameter is True, set the
        correlation coefficient threshold to define
        closely related columns. Defaults to 0.999

    Returns
    -------
    rankings : pd.DataFrame
        dataframe with three columns "rank", "feature", "entropy"
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
    rankings.set_index("rank", inplace=True)
    return rankings


def compute_entropy(dataframe: pd.DataFrame) -> float:
    """
    Function to carry out the mathematical calculations to calculate the
    entropy as defined in the research paper mentioned in the rank features
    function.
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
    """
    """

    correlation_matrix = dataframe.corr()
    correlated_columns = {}
    for col in correlation_matrix.columns:
        indices = correlation_matrix.index[
            abs(correlation_matrix[col]) > correlation_threshold
        ].tolist()
        correlated_columns[col] = indices
    return correlated_columns
