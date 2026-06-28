"""
LaborEcon — Labor Economics Toolkit

Core structural and reduced-form methods for labor economics research.

Modules
-------
mincer       : Mincer earnings equation
heckman      : Heckman selection model
decomposition: Oaxaca-Blinder and Juhn-Murphy-Pierce decompositions
search_match : Diamond-Mortensen-Pissarides search and matching model
quantile     : Quantile regression and wage decomposition
"""

from .mincer import MincerEquation
from .heckman import HeckmanSelection
from .decomposition import OaxacaBlinder
from .search_match import DiamondMortensenPissarides
from .quantile import QuantileRegression

__version__ = "0.1.0"

__all__ = [
    "MincerEquation",
    "HeckmanSelection",
    "OaxacaBlinder",
    "DiamondMortensenPissarides",
    "QuantileRegression",
]
