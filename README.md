# Feature Selection for Clustering: fselect

A fast and scalable implementation of A-RANK algorithm as proposed
by Dash, M. and Liu, H. in their paper "Feature Selection for Clustering" for selecting features
from a dataset using an entropy measure using fast python libraries: numpy, pandas and scikit-learn.

## Getting Started  

Install the package:

```  
pip install fselect
```

Import the main function:

```python
from fselect import rank_features  
```

Prepare a dataframe with normalized continuous features:  

```python  
import pandas as pd

df = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],    
    [...]
})
```

Rank the features:

```python
ranked_df = rank_features(df)  
```

The returned dataframe \`ranked_df\` contains columns: "rank", "feature", "entropy" sorted by entropy.

## Usage

The main parameters:  

- `dataframe: pd.DataFrame` - Input dataframe with continuous normalized features
- `remove_correlated_columns: bool` (optional) - Whether to remove highly correlated columns before ranking
- `correlation_threshold: float` (optional) - Correlation threshold to determine correlated columns (default 0.999)

**Remove correlated columns first**

```python
ranked_df = rank_features(df, remove_correlated_columns=True)  
```

**Custom correlation threshold**   

```python
ranked_df = rank_features(df, remove_correlated_columns=True, correlation_threshold=0.95) 
```

## Algorithm  

The entropy calculation is based on the equations defined in the ARANK paper. It calculates a similarity matrix of the dataframe and computes entropy from the same.
