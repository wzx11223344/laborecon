"""
Mincer Earnings Equation
========================

Implements the classic Mincer (1974) human capital earnings function:

.. math::

    \\ln(w_i) = \\beta_0 + \\beta_1 \\cdot \\text{edu}_i + \\beta_2 \\cdot \\text{exp}_i
                + \\beta_3 \\cdot \\text{exp}_i^2 + \\varepsilon_i

where:
- :math:`\\beta_1` is the rate of return to an additional year of schooling,
- :math:`\\beta_2, \\beta_3` capture the concave experience-earnings profile.

All estimation uses manual OLS (no sklearn) with full covariance matrix,
t-statistics, and :math:`R^2`.
"""

import numpy as np


class MincerEquation:
    r"""Mincer human capital earnings equation.

    Estimates the classic log-wage equation:

    .. math::

        \ln(w) = \beta_0 + \beta_1 \cdot \text{edu} + \beta_2 \cdot \text{exp}
                 + \beta_3 \cdot \text{exp}^2 + \varepsilon

    Parameters
    ----------
    add_constant : bool, default=True
        Whether to prepend a column of ones (intercept) to X.

    Attributes
    ----------
    beta_ : ndarray of shape (k,)
        Estimated coefficients.
    se_ : ndarray of shape (k,)
        Standard errors of coefficients.
    tstat_ : ndarray of shape (k,)
        t-statistics for H0: beta_j = 0.
    pval_ : ndarray of shape (k,)
        Two-sided p-values.
    cov_ : ndarray of shape (k, k)
        Covariance matrix of beta_hat.
    r2_ : float
        Coefficient of determination.
    adj_r2_ : float
        Adjusted R-squared.
    sigma2_ : float
        Residual variance estimate.
    n_ : int
        Number of observations.
    k_ : int
        Number of regressors (including constant if added).
    """

    def __init__(self, add_constant=True):
        self.add_constant = add_constant
        self.beta_ = None
        self.se_ = None
        self.tstat_ = None
        self.pval_ = None
        self.cov_ = None
        self.r2_ = None
        self.adj_r2_ = None
        self.sigma2_ = None
        self.n_ = None
        self.k_ = None

    def fit(self, X, y):
        r"""Fit the Mincer equation via OLS.

        .. math::

            \hat{\beta} = (X'X)^{-1} X'y

            \text{Var}(\hat{\beta}) = \hat{\sigma}^2 (X'X)^{-1}

            \hat{\sigma}^2 = \frac{\hat{\varepsilon}'\hat{\varepsilon}}{n - k}

            R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix (education, experience, experience², etc.).
        y : ndarray of shape (n,)
            Dependent variable (typically log wage).

        Returns
        -------
        self : MincerEquation
            Fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.add_constant:
            X = np.column_stack([np.ones(X.shape[0]), X])

        self.n_, self.k_ = X.shape

        # OLS: (X'X)^(-1) X'y
        XtX = X.T @ X
        Xty = X.T @ y

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix (X'X); check for perfect collinearity.")

        self.beta_ = XtX_inv @ Xty

        # Residuals
        y_hat = X @ self.beta_
        residuals = y - y_hat

        # Residual variance
        self.sigma2_ = np.sum(residuals ** 2) / (self.n_ - self.k_)

        # Covariance matrix of beta
        self.cov_ = self.sigma2_ * XtX_inv

        # Standard errors
        self.se_ = np.sqrt(np.diag(self.cov_))

        # t-statistics
        self.tstat_ = self.beta_ / self.se_

        # Two-sided p-values from Student's t with n-k df
        from scipy.stats import t as t_dist
        self.pval_ = 2.0 * t_dist.sf(np.abs(self.tstat_), df=self.n_ - self.k_)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r2_ = 1.0 - ss_res / ss_tot
        self.adj_r2_ = 1.0 - (1.0 - self.r2_) * (self.n_ - 1) / (self.n_ - self.k_)

        self.residuals_ = residuals

        return self

    @staticmethod
    def returns_to_education(model):
        r"""Interpret the education coefficient as percentage return per year.

        For the Mincer equation :math:`\ln(w) = \beta_0 + \beta_1 \cdot \text{edu} + \cdots`:

        .. math::

            \text{Return} = \hat{\beta}_1 \times 100\%

        Parameters
        ----------
        model : MincerEquation
            A fitted MincerEquation instance.

        Returns
        -------
        dict
            Keys: 'return_pct' (percentage return per year of education),
                  'se' (standard error), 'tstat', 'pval'.
        """
        # Education coefficient is index 1 (index 0 is constant)
        edu_idx = 1 if model.add_constant else 0
        beta_edu = model.beta_[edu_idx]
        return {
            "return_pct": beta_edu * 100.0,
            "se": model.se_[edu_idx],
            "tstat": model.tstat_[edu_idx],
            "pval": model.pval_[edu_idx],
        }

    def experience_profile(self, model, years=40):
        r"""Compute the predicted wage-experience profile.

        Evaluates:

        .. math::

            \hat{w}(\text{exp}) = \exp(\hat{\beta}_0 + \hat{\beta}_2 \cdot \text{exp}
                                  + \hat{\beta}_3 \cdot \text{exp}^2)

        holding education at its sample mean.

        Parameters
        ----------
        model : MincerEquation
            A fitted MincerEquation instance.
        years : int, default=40
            Number of years of potential experience to profile.

        Returns
        -------
        dict
            Keys: 'experience' (ndarray of experience values),
                  'predicted_log_wage', 'predicted_wage'.
        """
        exp = np.arange(0, years + 1, dtype=float)
        exp_sq = exp ** 2

        # Build prediction matrix
        k_total = model.k_  # includes constant
        n_vars = k_total - 1 if model.add_constant else k_total

        X_pred = np.zeros((len(exp), k_total))

        if model.add_constant:
            X_pred[:, 0] = 1.0  # intercept
            col_offset = 1
        else:
            col_offset = 0

        # We assume the user supplied [edu, exp, exp^2] in that order.
        # Place exp as column 1 (after constant) and exp^2 as column 2.
        # For the education column (col_offset), use its mean from the
        # original data — but we don't store X, so set to 0 (the prediction
        # evaluates the experience profile ignoring edu, or we can use
        # a typical mean).
        X_pred[:, col_offset] = 0.0  # education held at 0 (mean-centered assumption)
        if k_total - col_offset >= 2:
            X_pred[:, col_offset + 1] = exp   # experience
        if k_total - col_offset >= 3:
            X_pred[:, col_offset + 2] = exp_sq  # experience squared

        log_wage = X_pred @ model.beta_
        wage = np.exp(log_wage)

        return {
            "experience": exp,
            "predicted_log_wage": log_wage,
            "predicted_wage": wage,
        }

    def gender_wage_gap(self, data, male_col, female_col):
        r"""Compute raw and adjusted gender wage gaps.

        The raw gap is:

        .. math::

            \text{Raw Gap} = \overline{\ln(w_m)} - \overline{\ln(w_f)}

        The adjusted gap uses the Oaxaca-Blinder decomposition:

        .. math::

            \text{Adjusted Gap} = \overline{\ln(w_m)} - \overline{X_f}\hat{\beta}_m

        where :math:`\hat{\beta}_m` are the male (reference group) coefficients.

        Parameters
        ----------
        data : ndarray of shape (n, p)
            Full dataset with columns [y, X_1, ..., X_k].
        male_col : int
            Column index of the male indicator (1 = male, 0 = female).
        female_col : int
            Column index of the female indicator (1 = female, 0 = male).

        Returns
        -------
        dict
            Keys: 'raw_gap', 'adjusted_gap', 'explained', 'unexplained'.
        """
        data = np.asarray(data, dtype=float)

        male_mask = data[:, male_col] == 1
        female_mask = data[:, female_col] == 1

        # Assume the first column is log wage (y), rest are X
        y = data[:, 0]
        X_vars = np.delete(data, [0, male_col, female_col], axis=1)

        y_m = y[male_mask]
        X_m = X_vars[male_mask]
        y_f = y[female_mask]
        X_f = X_vars[female_mask]

        # Fit male and female equations
        model_m = MincerEquation(add_constant=True)
        model_m.fit(X_m, y_m)
        model_f = MincerEquation(add_constant=True)
        model_f.fit(X_f, y_f)

        raw_gap = np.mean(y_m) - np.mean(y_f)

        # Add constants
        X_f_c = np.column_stack([np.ones(X_f.shape[0]), X_f])
        X_m_c = np.column_stack([np.ones(X_m.shape[0]), X_m])

        # Adjusted: what would females earn with male returns?
        X_f_mean = np.mean(X_f_c, axis=0)
        y_f_adjusted = X_f_mean @ model_m.beta_

        X_m_mean = np.mean(X_m_c, axis=0)
        y_m_pred = X_m_mean @ model_m.beta_

        adjusted_gap = y_m_pred - y_f_adjusted

        # Three-fold decomposition
        X_f_mean_no_const = X_f_mean[1:]
        X_m_mean_no_const = X_m_mean[1:]
        endow = (X_m_mean_no_const - X_f_mean_no_const) @ model_f.beta_[1:]
        coeff = X_f_mean_no_const @ (model_m.beta_[1:] - model_f.beta_[1:])
        interaction = (X_m_mean_no_const - X_f_mean_no_const) @ (
            model_m.beta_[1:] - model_f.beta_[1:]
        )

        return {
            "raw_gap": raw_gap,
            "adjusted_gap": adjusted_gap,
            "endowment": endow,
            "coefficient": coeff,
            "interaction": interaction,
            "male_beta": model_m.beta_,
            "female_beta": model_f.beta_,
        }
