# LaborEcon — Labor Economics Toolkit

A pure NumPy toolkit for core structural and reduced-form methods in labor economics.

## Modules

| Module | Description |
|---|---|
| `mincer.py` | Mincer earnings equation (OLS, returns to education, experience profiles, gender wage gap) |
| `heckman.py` | Heckman selection model (two-step estimator with standard error correction, full MLE) |
| `decomposition.py` | Oaxaca-Blinder decomposition (three-fold), Juhn-Murphy-Pierce decomposition |
| `search_match.py` | Diamond-Mortensen-Pissarides search and matching model (steady state, comparative statics, Beveridge curve) |
| `quantile.py` | Quantile regression (simplex-based LP), Machado-Mata decomposition, wage inequality trends |

## Installation

```bash
pip install -e .
```

Dependencies: `numpy` only.

## Quick Start

```python
import numpy as np
from laborecon.mincer import MincerEquation
from laborecon.heckman import HeckmanSelection
from laborecon.decomposition import OaxacaBlinder
from laborecon.search_match import DiamondMortensenPissarides

# Mincer equation
X = np.column_stack([edu, exp, exp**2])
y = np.log(wage)
model = MincerEquation()
results = model.fit(X, y)
print(model.returns_to_education(results))  # % return per year of education

# Heckman selection correction
heckman = HeckmanSelection()
results = heckman.heckman_two_step(y, X, Z, D)
print(heckman.selection_correction(results))

# Oaxaca-Blinder decomposition
ob = OaxacaBlinder()
decomp = ob.oaxaca_blinder(y_male, X_male, y_female, X_female)
print(decomp["endowment"], decomp["coefficient"], decomp["interaction"])

# DMP search model
dmp = DiamondMortensenPissarides()
ss = dmp.steady_state(params)
print(ss["theta"], ss["u"], ss["w"])
```

## License

MIT
