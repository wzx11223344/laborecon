"""
Wage Decomposition Methods
==========================

Implements classic wage gap decompositions:

Oaxaca-Blinder (1973) three-fold decomposition:
.. math::

    \\Delta \\bar{y} = \\underbrace{(\\bar{X}_A - \\bar{X}_B)'\\hat{\\beta}_B}_{\\text{Endowment}}
                     + \\underbrace{\\bar{X}_B'(\\hat{\\beta}_A - \\hat{\\beta}_B)}_{\\text{Coefficient}}
                     + \\underbrace{(\\bar{X}_A - \\bar{X}_B)'(\\hat{\\beta}_A -
                       \\hat{\\beta}_B)}_{\\text{Interaction}}

Juhn-Murphy-Pierce (1991) decomposition:
.. math::

    y_A(\\theta) - y_B(\\theta) = \\underbrace{(X_A(\\theta) - X_B(\\theta))'\\beta_B
                                 + X_A(\\theta)'(\\beta_A - \\beta_B)}_{\\text{Characteristics \& Prices}}
                                 + \\underbrace{(F_A^{-1}(\\theta) -
                                 F_B^{-1}(\\theta))\\sigma_B
                                 + F_A^{-1}(\\theta)(\\sigma_A - \\sigma_B)}_{\\text{Residual}}
"""

import numpy as np


class OaxacaBlinder:
    r"""Oaxaca-Blinder wage gap decomposition.

    Decomposes the mean wage differential between two groups (e.g., male/female,
    white/black) into endowment, coefficient, and interaction effects.

    Parameters
    ----------
    add_constant : bool, default=True
        Whether to add an intercept to the regressor matrices.

    Attributes
    ----------
    beta_A_ : ndarray
        OLS coefficients for group A.
    beta_B_ : ndarray
        OLS coefficients for group B.
    """

    def __init__(self, add_constant=True):
        self.add_constant = add_constant
        self.beta_A_ = None
        self.beta_B_ = None

    @staticmethod
    def _ols(X, y):
        """Manual OLS: (X'X)^(-1) X'y."""
        XtX = X.T @ X
        Xty = X.T @ y
        return np.linalg.solve(XtX, Xty)

    def oaxaca_blinder(self, y_A, X_A, y_B, X_B):
        r"""Oaxaca-Blinder three-fold decomposition.

        .. math::

            \\Delta\\bar{y} = \\underbrace{(\\bar{X}_A - \\bar{X}_B)'\\hat{\\beta}_B}_{E}
                            + \\underbrace{\\bar{X}_B'(\\hat{\\beta}_A - \\hat{\\beta}_B)}_{C}
                            + \\underbrace{(\\bar{X}_A - \\bar{X}_B)'(\\hat{\\beta}_A -
                              \\hat{\\beta}_B)}_{I}

        where :math:`E` = Endowment (explained), :math:`C` = Coefficient (unexplained),
        :math:`I` = Interaction.

        Parameters
        ----------
        y_A : ndarray of shape (n_A,)
            Outcome for group A.
        X_A : ndarray of shape (n_A, k)
            Covariates for group A.
        y_B : ndarray of shape (n_B,)
            Outcome for group B.
        X_B : ndarray of shape (n_B, k)
            Covariates for group B.

        Returns
        -------
        dict
            Keys: 'raw_gap', 'endowment', 'coefficient', 'interaction',
                  'beta_A', 'beta_B', 'X_mean_A', 'X_mean_B'.
        """
        X_A = np.asarray(X_A, dtype=float)
        X_B = np.asarray(X_B, dtype=float)
        y_A = np.asarray(y_A, dtype=float).ravel()
        y_B = np.asarray(y_B, dtype=float).ravel()

        if self.add_constant:
            X_A = np.column_stack([np.ones(X_A.shape[0]), X_A])
            X_B = np.column_stack([np.ones(X_B.shape[0]), X_B])

        # OLS estimation
        self.beta_A_ = self._ols(X_A, y_A)
        self.beta_B_ = self._ols(X_B, y_B)

        # Group means
        X_mean_A = np.mean(X_A, axis=0)
        X_mean_B = np.mean(X_B, axis=0)

        # Raw gap
        raw_gap = np.mean(y_A) - np.mean(y_B)

        # Three-fold decomposition
        dX = X_mean_A - X_mean_B
        dB = self.beta_A_ - self.beta_B_

        endowment = dX @ self.beta_B_
        coefficient = X_mean_B @ dB
        interaction = dX @ dB

        return {
            "raw_gap": raw_gap,
            "endowment": endowment,
            "coefficient": coefficient,
            "interaction": interaction,
            "beta_A": self.beta_A_,
            "beta_B": self.beta_B_,
            "X_mean_A": X_mean_A,
            "X_mean_B": X_mean_B,
            "check_sum": endowment + coefficient + interaction,
        }

    def detailed_decomposition(self, model, var_names=None):
        r"""Per-variable contributions to the Oaxaca-Blinder decomposition.

        For each variable :math:`j`:

        .. math::

            E_j = (\\bar{X}_{Aj} - \\bar{X}_{Bj}) \\cdot \\hat{\\beta}_{Bj}

            C_j = \\bar{X}_{Bj} \\cdot (\\hat{\\beta}_{Aj} - \\hat{\\beta}_{Bj})

            I_j = (\\bar{X}_{Aj} - \\bar{X}_{Bj}) \\cdot (\\hat{\\beta}_{Aj}
                  - \\hat{\\beta}_{Bj})

        Parameters
        ----------
        model : dict
            Output from `oaxaca_blinder`.
        var_names : list of str, optional
            Variable names for display. Must match the number of covariates
            (including constant if add_constant=True).

        Returns
        -------
        dict
            Keys: 'endowment' (per-variable), 'coefficient' (per-variable),
                  'interaction' (per-variable), 'names'.
        """
        dX = model["X_mean_A"] - model["X_mean_B"]
        dB = model["beta_A"] - model["beta_B"]
        X_mean_B = model["X_mean_B"]

        k = len(dX)
        if var_names is None:
            has_const = self.add_constant
            if has_const:
                var_names = ["const"] + [f"X{j}" for j in range(1, k)]
            else:
                var_names = [f"X{j}" for j in range(k)]

        endow_detail = dX * model["beta_B"]
        coeff_detail = X_mean_B * dB
        inter_detail = dX * dB

        return {
            "endowment": {var_names[j]: endow_detail[j] for j in range(k)},
            "coefficient": {var_names[j]: coeff_detail[j] for j in range(k)},
            "interaction": {var_names[j]: inter_detail[j] for j in range(k)},
            "names": var_names,
        }

    def counterfactual_distribution(self, y_A, X_A, y_B, X_B):
        r"""Construct counterfactual wage distributions.

        Computes two counterfactuals:

        .. math::

            y_A^{\\text{cf}} = X_A\\hat{\\beta}_B \\quad
            \\text{(Group A characteristics with Group B returns)}

            y_B^{\\text{cf}} = X_B\\hat{\\beta}_A \\quad
            \\text{(Group B characteristics with Group A returns)}

        Parameters
        ----------
        y_A : ndarray
            Group A outcomes.
        X_A : ndarray
            Group A covariates.
        y_B : ndarray
            Group B outcomes.
        X_B : ndarray
            Group B covariates.

        Returns
        -------
        dict
            Keys: 'y_A_actual', 'y_B_actual', 'y_A_counterfactual',
                  'y_B_counterfactual'.
        """
        X_A = np.asarray(X_A, dtype=float)
        X_B = np.asarray(X_B, dtype=float)

        if self.add_constant:
            X_A = np.column_stack([np.ones(X_A.shape[0]), X_A])
            X_B = np.column_stack([np.ones(X_B.shape[0]), X_B])

        if self.beta_A_ is None or self.beta_B_ is None:
            y_A_arr = np.asarray(y_A, dtype=float).ravel()
            y_B_arr = np.asarray(y_B, dtype=float).ravel()
            self.beta_A_ = self._ols(X_A, y_A_arr)
            self.beta_B_ = self._ols(X_B, y_B_arr)

        y_A_cf = X_A @ self.beta_B_
        y_B_cf = X_B @ self.beta_A_

        return {
            "y_A_actual": y_A,
            "y_B_actual": y_B,
            "y_A_counterfactual": y_A_cf,
            "y_B_counterfactual": y_B_cf,
        }

    def juhn_murphy_pierce(self, y_A, X_A, y_B, X_B, n_quantiles=100):
        r"""Juhn-Murphy-Pierce (1991) decomposition incorporating
        the full residual distribution.

        The JMP decomposition extends Oaxaca-Blinder by accounting for
        differences in the residual (unobservable) distribution at every
        quantile.

        Procedure:
        1. Estimate :math:`\\hat{\\beta}_A`, :math:`\\hat{\\beta}_B`
        2. Compute residuals :math:`\\varepsilon_A = y_A - X_A\\hat{\\beta}_A`
        3. Compute residual standard deviations :math:`\\sigma_A`, :math:`\\sigma_B`
        4. Standardize: :math:`\\theta_i = \\varepsilon_{Ai} / \\sigma_A`
        5. Counterfactual: :math:`y_i^{\\text{cf}} = X_{Ai}\\hat{\\beta}_B
           + \\sigma_B \\cdot \\theta_i`

        Parameters
        ----------
        y_A : ndarray
            Group A outcomes.
        X_A : ndarray
            Group A covariates.
        y_B : ndarray
            Group B outcomes.
        X_B : ndarray
            Group B covariates.
        n_quantiles : int, default=100
            Number of quantiles for distribution comparison.

        Returns
        -------
        dict
            Keys: 'quantiles', 'gap', 'quantity_effect', 'price_effect',
                  'residual_effect', 'sigma_A', 'sigma_B', 'beta_A', 'beta_B'.
        """
        X_A = np.asarray(X_A, dtype=float)
        X_B = np.asarray(X_B, dtype=float)
        y_A = np.asarray(y_A, dtype=float).ravel()
        y_B = np.asarray(y_B, dtype=float).ravel()

        nA = len(y_A)
        nB = len(y_B)

        if self.add_constant:
            X_A = np.column_stack([np.ones(nA), X_A])
            X_B = np.column_stack([np.ones(nB), X_B])

        # Estimate wage equations
        beta_A = self._ols(X_A, y_A)
        beta_B = self._ols(X_B, y_B)

        # Residuals
        eps_A = y_A - X_A @ beta_A
        eps_B = y_B - X_B @ beta_B

        # Residual standard deviations
        sigma_A = np.std(eps_A, ddof=1)
        sigma_B = np.std(eps_B, ddof=1)

        # Standardized residuals (percentile ranks for group A)
        eps_A_std = eps_A / (sigma_A if sigma_A > 1e-12 else 1.0)
        ranks_A = np.argsort(np.argsort(eps_A_std)) / (nA - 1)

        # Counterfactual residuals for A: use B's sigma but A's rank
        eps_cf = sigma_B * eps_A_std

        # Counterfactual wages
        y_A_cf = X_A @ beta_B + eps_cf

        # Quantile analysis
        quantiles = np.linspace(1, 99, n_quantiles) / 100.0
        q_A = np.quantile(y_A, quantiles)
        q_B = np.quantile(y_B, quantiles)
        q_cf = np.quantile(y_A_cf, quantiles)

        gap = q_A - q_B
        # Characteristics + price effect (gap - residual effect)
        quantity_price = q_A - q_cf

        # Approximate: compute X_A quantiles
        y_pred_A = X_A @ beta_A
        y_pred_B_cf = X_A @ beta_B  # A characteristics with B prices

        # Separate the components more precisely
        # Quantity (characteristics) effect at each quantile
        X_q_A = X_A[np.argsort(y_A)]
        X_q_B = X_B[np.argsort(y_B)]
        y_pred_q = X_q_A @ beta_A

        # Price effect: X_A' * (beta_A - beta_B)
        price_effect = X_A @ (beta_A - beta_B)
        q_price = np.quantile(price_effect, quantiles)

        # Residual effect
        residual_effect = gap - (q_A - q_cf)

        return {
            "quantiles": quantiles,
            "q_A": q_A,
            "q_B": q_B,
            "q_counterfactual": q_cf,
            "gap": gap,
            "quantity_price_effect": quantity_price,
            "residual_effect": residual_effect,
            "sigma_A": sigma_A,
            "sigma_B": sigma_B,
            "beta_A": beta_A,
            "beta_B": beta_B,
        }
