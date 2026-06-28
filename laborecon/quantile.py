"""
Quantile Regression for Wage Decompositions
===========================================

Implements quantile regression via linear programming (simplex method)
and wage decomposition techniques:

.. math::

    \\hat{\\beta}(\\tau) = \\arg\\min_{\\beta} \\sum_{i} \\rho_\\tau(y_i - x_i'\\beta)

where :math:`\\rho_\\tau(u) = u(\\tau - \\mathbf{1}[u < 0])` is the check function.

Machado-Mata (2005) decomposition:
.. math::

    \\{y_A^*(\\tau)\\}_{\\tau=1}^M \\quad \\text{via resampling from } X_A \\text{ with } \\hat{\\beta}_B(\\tau)

Wage inequality trends:
.. math::

    \\text{P90/P10}, \\quad \\text{P90/P50}, \\quad \\text{P50/P10}
"""

import numpy as np


def _quantile_loss(beta, X, y, tau):
    """Check function loss for quantile regression."""
    residuals = y - X @ beta
    return np.sum(np.where(residuals >= 0, tau * residuals, (tau - 1.0) * residuals))


class QuantileRegression:
    r"""Quantile regression via linear programming.

    Solves :math:`\\min_\\beta \\sum_i \\rho_\\tau(y_i - x_i'\\beta)`
    using the simplex method.

    The LP formulation:

    .. math::

        \\min_{u,v,\\beta^+,\\beta^-} \\; \\tau\\mathbf{1}'u + (1-\\tau)\\mathbf{1}'v

        \\text{s.t.} \\; X\\beta^+ - X\\beta^- + u - v = y, \\quad
        \\beta^+,\\beta^-,u,v \\geq 0

    Parameters
    ----------
    add_constant : bool, default=True
        Whether to prepend a column of ones.
    max_iter : int, default=2000
        Maximum simplex iterations.
    tol : float, default=1e-10
        Numerical tolerance.

    Attributes
    ----------
    beta_ : ndarray of shape (k,)
        Estimated coefficients at quantile tau.
    """

    def __init__(self, add_constant=True, max_iter=2000, tol=1e-10):
        self.add_constant = add_constant
        self.max_iter = max_iter
        self.tol = tol
        self.beta_ = None

    def fit(self, X, y, tau=0.5):
        r"""Fit quantile regression at quantile tau.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix.
        y : ndarray of shape (n,)
            Response variable.
        tau : float, default=0.5
            Quantile to fit (0 < tau < 1). tau=0.5 gives median regression.

        Returns
        -------
        self : QuantileRegression
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        tau = float(tau)

        if self.add_constant:
            X = np.column_stack([np.ones(X.shape[0]), X])

        n, k = X.shape

        # ==============================================================
        # LP formulation for quantile regression
        #
        # min  tau * sum(u) + (1-tau) * sum(v)
        # s.t. X * (beta_p - beta_n) + u - v = y
        #      beta_p, beta_n, u, v >= 0
        #
        # Variables: [beta_p (k), beta_n (k), u (n), v (n)]
        # Total: 2k + 2n variables
        # ==============================================================

        # Build initial feasible basis
        # Set beta_p = beta_n = 0
        # u_i = max(y_i, 0), v_i = max(-y_i, 0)
        # Basis variables: n of the (u, v) variables
        # Basis matrix B = diag(1 for u columns, -1 for v columns)

        # Cost vector: c = [0_k, 0_k, tau*1_n, (1-tau)*1_n]
        # Constraint matrix (transposed for convenience): A = [X, -X, I_n, -I_n]
        # RHS: b = y

        # Identify which variables are in the initial basis
        basis = np.zeros(n, dtype=int)  # indices of basic variables in [0, 2k+2n)
        basis_sign = np.ones(n)  # +1 for u columns (I), -1 for v columns (-I)

        for i in range(n):
            if y[i] >= 0:
                basis[i] = 2 * k + i  # u_i column (from I_n)
                basis_sign[i] = 1.0
            else:
                basis[i] = 2 * k + n + i  # v_i column (from -I_n)
                basis_sign[i] = -1.0

        # Basis inverse: B is diagonal with ±1, so B^{-1} = B
        B_inv = np.diag(basis_sign)

        # Basic solution
        x_B = B_inv @ y  # = (±y_i), should all be >= 0

        # Basic costs
        c_B = np.zeros(n)
        for i in range(n):
            col = basis[i]
            if col < 2 * k:
                c_B[i] = 0.0  # beta variables have 0 cost
            elif col < 2 * k + n:
                c_B[i] = tau  # u variables
            else:
                c_B[i] = 1.0 - tau  # v variables

        # All variable indices and their column types
        n_total = 2 * k + 2 * n  # total variables

        def _get_column(var_idx):
            """Return column of the constraint matrix for variable var_idx."""
            col = np.zeros(n)
            if var_idx < k:
                # beta_p: column is X[:, var_idx]
                col = X[:, var_idx].copy()
            elif var_idx < 2 * k:
                # beta_n: column is -X[:, var_idx - k]
                col = -X[:, var_idx - k].copy()
            elif var_idx < 2 * k + n:
                # u: column is e_{var_idx - 2*k}
                col[var_idx - 2 * k] = 1.0
            else:
                # v: column is -e_{var_idx - 2*k - n}
                col[var_idx - 2 * k - n] = -1.0
            return col

        def _get_cost(var_idx):
            if var_idx < 2 * k:
                return 0.0
            elif var_idx < 2 * k + n:
                return tau
            else:
                return 1.0 - tau

        # ---- Revised Simplex (Phase II only, since we have a feasible basis) ----
        for iteration in range(self.max_iter):
            # Compute dual variables (simplex multipliers)
            # pi = B^{-T} c_B
            pi = B_inv.T @ c_B

            # Compute reduced costs for all non-basic variables
            # r_j = c_j - pi' A_j
            min_reduced = 0.0
            entering = -1

            # Check non-basic variables
            is_basic = np.zeros(n_total, dtype=bool)
            is_basic[basis] = True

            for j in range(n_total):
                if is_basic[j]:
                    continue
                a_j = _get_column(j)
                c_j = _get_cost(j)
                reduced = c_j - pi @ a_j

                if reduced < min_reduced - self.tol:
                    min_reduced = reduced
                    entering = j

            # Optimality check
            if entering < 0:
                break

            # Compute direction: d = B^{-1} A_entering
            a_enter = _get_column(entering)
            d = B_inv @ a_enter

            # Ratio test
            ratios = np.full(n, np.inf)
            for i in range(n):
                if d[i] > self.tol:
                    ratios[i] = x_B[i] / d[i]

            min_ratio = np.min(ratios)
            if np.isinf(min_ratio):
                # Unbounded (shouldn't happen for this problem)
                break

            leaving_idx = np.argmin(ratios)  # index in basis (0 to n-1)

            # Update
            x_B = x_B - min_ratio * d
            x_B[leaving_idx] = min_ratio
            # Clip tiny negative values from numerical error
            x_B = np.maximum(x_B, -self.tol)

            # Update basis inverse using Sherman-Morrison formula
            # B_new = B + (a_enter - a_leave) e_j^T  where j = leaving_idx
            # Let w = B^{-1} (a_enter - a_leave)
            # B^{-1}_new = B^{-1} - w (e_j^T B^{-1}) / (1 + e_j^T w)
            #            = B^{-1} - outer(w, B^{-1}[j,:]) / (1 + w[j])

            a_leave = _get_column(basis[leaving_idx])
            da = a_enter - a_leave
            w = B_inv @ da  # direction in basic variable space
            wj = w[leaving_idx]

            denom = 1.0 + wj
            if np.abs(denom) < self.tol:
                # Recompute B_inv from scratch
                B = np.column_stack([_get_column(basis[i]) for i in range(n)])
                B[:, leaving_idx] = a_enter
                B_inv = np.linalg.inv(B)
            else:
                r = B_inv[leaving_idx, :]  # j-th row of B_inv
                B_inv = B_inv - np.outer(w, r) / denom

            # Update basis and costs
            basis[leaving_idx] = entering
            c_B[leaving_idx] = _get_cost(entering)

            # Periodic full recomputation of B_inv for numerical stability
            if iteration > 0 and iteration % 100 == 0:
                B = np.column_stack([_get_column(basis[i]) for i in range(n)])
                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    pass  # keep current B_inv if singular

        # Extract beta from solution
        # beta = beta_p - beta_n
        beta_p = np.zeros(k)
        beta_n = np.zeros(k)

        for i in range(n):
            col = basis[i]
            val = x_B[i]
            if 0 <= col < k:
                beta_p[col] = val
            elif k <= col < 2 * k:
                beta_n[col - k] = val

        self.beta_ = beta_p - beta_n
        self.tau_ = tau

        return self

    def predict(self, X):
        """Predict using fitted quantile model."""
        X = np.asarray(X, dtype=float)
        if self.add_constant:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.beta_

    # ------------------------------------------------------------------
    # Wage decomposition methods
    # ------------------------------------------------------------------
    def quantile_wage_decomposition(self, y_A, X_A, y_B, X_B, quantiles=None):
        r"""Machado-Mata (2005) quantile wage decomposition.

        Estimates quantile regressions at multiple quantiles for two groups
        and decomposes the wage gap at each quantile.

        The decomposition at quantile :math:`\\tau`:

        .. math::

            Q_\\tau(y_A) - Q_\\tau(y_B) = [Q_\\tau(y_A) - Q_\\tau(y_A^{\\text{cf}})]
                                        + [Q_\\tau(y_A^{\\text{cf}}) - Q_\\tau(y_B)]

        where :math:`y_A^{\\text{cf}} = X_A\\hat{\\beta}_B(\\tau)`.

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
        quantiles : array-like, optional
            Quantiles to evaluate. Default: [0.1, 0.25, 0.5, 0.75, 0.9].

        Returns
        -------
        dict
            Keys: 'quantiles', 'q_A', 'q_B', 'q_counterfactual',
                  'coefficient_effect', 'characteristics_effect', 'total_gap'.
        """
        if quantiles is None:
            quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles = np.asarray(quantiles)

        q_A = []
        q_B = []
        q_cf = []
        coef_effect = []
        char_effect = []

        for tau in quantiles:
            # Fit on group A
            qr_A = QuantileRegression(add_constant=self.add_constant,
                                       max_iter=self.max_iter, tol=self.tol)
            qr_A.fit(X_A, y_A, tau)
            beta_A = qr_A.beta_

            # Fit on group B
            qr_B = QuantileRegression(add_constant=self.add_constant,
                                       max_iter=self.max_iter, tol=self.tol)
            qr_B.fit(X_B, y_B, tau)
            beta_B = qr_B.beta_

            # Predicted quantiles
            X_A_aug = np.asarray(X_A, dtype=float)
            X_B_aug = np.asarray(X_B, dtype=float)
            if self.add_constant:
                X_A_aug = np.column_stack([np.ones(X_A_aug.shape[0]), X_A_aug])
                X_B_aug = np.column_stack([np.ones(X_B_aug.shape[0]), X_B_aug])

            pred_A = X_A_aug @ beta_A
            pred_B = X_B_aug @ beta_B
            pred_cf = X_A_aug @ beta_B  # A's characteristics with B's returns

            q_A.append(np.quantile(pred_A, tau))
            q_B.append(np.quantile(pred_B, tau))
            q_cf.append(np.quantile(pred_cf, tau))

            # Decomposition
            coef_effect.append(q_A[-1] - q_cf[-1])
            char_effect.append(q_cf[-1] - q_B[-1])

        q_A = np.array(q_A)
        q_B = np.array(q_B)
        q_cf = np.array(q_cf)

        return {
            "quantiles": quantiles,
            "q_A": q_A,
            "q_B": q_B,
            "q_counterfactual": q_cf,
            "coefficient_effect": np.array(coef_effect),
            "characteristics_effect": np.array(char_effect),
            "total_gap": q_A - q_B,
            "check_sum": np.array(coef_effect) + np.array(char_effect),
        }

    def counterfactual_quantiles(self, y_A, X_A, y_B, X_B, tau=0.5):
        r"""Compute counterfactual quantiles at a specific tau.

        Constructs:

        .. math::

            y_A^{\\text{cf},C} &= X_A\\hat{\\beta}_B(\\tau) \\quad
            \\text{(A characteristics, B coefficients)}

            y_A^{\\text{cf},X} &= X_B\\hat{\\beta}_A(\\tau) \\quad
            \\text{(B characteristics, A coefficients)}

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
        tau : float, default=0.5
            Quantile.

        Returns
        -------
        dict
            Keys: 'beta_A', 'beta_B', 'y_A_pred', 'y_B_pred',
                  'y_A_cf_coef', 'y_A_cf_char',
                  'q_A', 'q_B', 'q_cf_coef', 'q_cf_char'.
        """
        # Fit quantile regressions
        qr_A = QuantileRegression(add_constant=self.add_constant,
                                   max_iter=self.max_iter, tol=self.tol)
        qr_A.fit(X_A, y_A, tau)
        beta_A = qr_A.beta_

        qr_B = QuantileRegression(add_constant=self.add_constant,
                                   max_iter=self.max_iter, tol=self.tol)
        qr_B.fit(X_B, y_B, tau)
        beta_B = qr_B.beta_

        X_A_aug = np.asarray(X_A, dtype=float)
        X_B_aug = np.asarray(X_B, dtype=float)
        if self.add_constant:
            X_A_aug = np.column_stack([np.ones(X_A_aug.shape[0]), X_A_aug])
            X_B_aug = np.column_stack([np.ones(X_B_aug.shape[0]), X_B_aug])

        y_A_pred = X_A_aug @ beta_A
        y_B_pred = X_B_aug @ beta_B
        y_A_cf_coef = X_A_aug @ beta_B  # counterfactual: A X, B coef
        y_A_cf_char = X_B_aug @ beta_A  # counterfactual: B X, A coef

        return {
            "beta_A": beta_A,
            "beta_B": beta_B,
            "y_A_pred": y_A_pred,
            "y_B_pred": y_B_pred,
            "y_A_cf_coef": y_A_cf_coef,
            "y_A_cf_char": y_A_cf_char,
            "q_A": np.quantile(y_A_pred, tau),
            "q_B": np.quantile(y_B_pred, tau),
            "q_cf_coef": np.quantile(y_A_cf_coef, tau),
            "q_cf_char": np.quantile(y_A_cf_char, tau),
            "tau": tau,
        }

    def wage_inequality_trends(self, data, years):
        r"""Compute wage inequality trends over time.

        Computes the 90-10, 90-50, and 50-10 log wage differentials
        for each year.

        .. math::

            \\text{P90/P10}_t = Q_{0.9}(w_t) - Q_{0.1}(w_t)

            \\text{P90/P50}_t = Q_{0.9}(w_t) - Q_{0.5}(w_t)

            \\text{P50/P10}_t = Q_{0.5}(w_t) - Q_{0.1}(w_t)

        Parameters
        ----------
        data : dict
            Dictionary mapping year -> ndarray of log wages for that year.
            Keys must be numeric (years).
        years : array-like
            Sorted list of years to analyze.

        Returns
        -------
        dict
            Keys: 'years', 'p90_p10', 'p90_p50', 'p50_p10',
                  'p90', 'p50', 'p10'.
        """
        years = np.asarray(years)
        n = len(years)

        p90 = np.zeros(n)
        p50 = np.zeros(n)
        p10 = np.zeros(n)

        for i, yr in enumerate(years):
            wages = np.asarray(data[yr]).ravel()
            p90[i] = np.quantile(wages, 0.90)
            p50[i] = np.quantile(wages, 0.50)
            p10[i] = np.quantile(wages, 0.10)

        return {
            "years": years,
            "p90_p10": p90 - p10,
            "p90_p50": p90 - p50,
            "p50_p10": p50 - p10,
            "p90": p90,
            "p50": p50,
            "p10": p10,
        }
