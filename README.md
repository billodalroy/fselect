# Feature Selection for Clustering: fselect

[![CI](https://img.shields.io/github/actions/workflow/status/billodalroy/fselect/ci.yml?branch=main&style=flat-square&label=tests&logo=github)](https://github.com/billodalroy/fselect/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/fselect?color=green&style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/fselect/)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/fselect/)
[![License](https://img.shields.io/github/license/billodalroy/fselect?color=blue&style=flat-square)](https://github.com/billodalroy/fselect/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/fselect)](https://pepy.tech/project/fselect)
[![Monthly downloads](https://static.pepy.tech/badge/fselect/month)](https://pepy.tech/project/fselect)

A fast and scalable implementation of A-RANK algorithm as proposed
by Dash, M. and Liu, H. in their paper "Feature Selection for Clustering" for selecting features
from a dataset using an entropy measure using fast python libraries: numpy, pandas and scikit-learn.

## Getting Started  

Install the package:

```bash
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

## Development

The project targets Python 3.10–3.14 and uses [uv](https://docs.astral.sh/uv/).

```bash
uv venv            # create the virtual environment
uv sync            # install runtime + dev (pytest) dependencies
uv run pytest      # run the test suite
uv build           # build the sdist + wheel (hatchling backend)
```

Run the suite against a specific interpreter (uv fetches it if needed):

```bash
uv run --python 3.10 pytest
```
