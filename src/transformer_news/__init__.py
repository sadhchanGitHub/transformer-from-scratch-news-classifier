"""
Initializes the `transformer_news` package.

The `transformer_news` package contains all the source code required to build,
train, and run the Transformer-based model for news classification. It is
structured into several modules, each responsible for a specific part of the
machine learning workflow.

This `__init__.py` file serves two main functions:
1.  It marks the `src/transformer_news` directory as a Python package, which
    allows for the use of relative imports within the project (e.g.,
    `from . import config`).
2.  It defines the public API of the package through the `__all__` variable.
    This explicitly lists which modules should be imported when a wildcard
    import (`from transformer_news import *`) is used, and it helps tools
    and developers understand which modules are intended for public use.

By importing the modules here, they can be accessed more conveniently. For
example, one can now use `transformer_news.model` instead of the more verbose
`transformer_news.model.model`.

Attributes:
    config (module): The module containing all project configurations and hyperparameters.
    model (module): The module defining the TransformerClassifier neural network architecture.
    utils (module): The module providing helper functions, such as the evaluation metric.
"""

from . import config
from . import model
from . import utils

# Defines the public API of the package. When a user writes
# `from transformer_news import *`, only the names in this list will be imported.
__all__ = ["config", "model", "utils"]