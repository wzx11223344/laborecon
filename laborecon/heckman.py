"""
Heckman Selection Model
=======================

Implements the Heckman (1979) sample selection correction model.

Selection equation (Probit):
.. math::

    D_i^* = Z_i \\gamma + u_i, \\quad D_i = \\mathbf{1}[D_i^* > 0]

Outcome equation:
.. math::

    y_i = X_i \\beta + \\varepsilon_i, \\quad
    \\begin{pmatrix} u_i \\\\ \\varepsilon_i \\end{pmatrix}
    \\sim \\mathcal{N}\\left(\\begin{pmatrix}0\\\\0\\end{pmatrix},
    \\begin{pmatrix}1 & \\rho\\sigma \\\\ \\rho\\sigma & \\sigma^2\\end{pmatrix}\\right)

where :math:`y_i` is observed only when :math:`D_i = 1`.

Two-step estimator (Heckman 1979):
1. Probit on selection: :math:`\\hat{\\gamma}` from :math:`P(D=1|Z) = \\Phi(Z\\gamma)`
2. OLS with Inverse Mills Ratio correction:
   :math:`y_i = X_i\\beta + \\rho\\sigma \\cdot \\lambda(Z_i\\hat{\\gamma}) + \\eta_i`

where :math:`\\lambda(\\cdot) = \\phi(\\cdot)/\\Phi(\\cdot)` is the Inverse Mills Ratio.
"""

import numpy as np


def _norm_pdf(x):
    """Standard normal PDF: phi(x)."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x):
    """Standard normal CDF: Phi(x).

    Uses the Abramowitz & Stegun approximation (error < 7.5e-8).
    """
    # Rational approximation to the normal CDF
    c = np.array([0.319381530, -0.356563782, 1.781477937,
                  -1.821255978, 1.330274429])
    p = 0.2316419
    sgn = np.where(x >= 0, 1.0, -1.0)
    t = 1.0 / (1.0 + p * np.abs(x))
    poly = c[0] * t + c[1] * t ** 2 + c[2] * t ** 3 + c[3] * t ** 4 + c[4] * t ** 5
    phi = _norm_pdf(x)
    return np.where(x >= 0, 1.0 - phi * poly, phi * poly)


def _inverse_mills_ratio(z):
    r"""Inverse Mills Ratio: :math:`\\lambda(z) = \\phi(z) / \\Phi(z)`.

    Parameters
    ----------
    z : ndarray
        Linear index Z*gamma.

    Returns
    -------
    ndarray
        Inverse Mills Ratio evaluated at each z.
    """
    phi = _norm_pdf(z)
    Phi = _norm_cdf(z)
    # Avoid division by zero
    Phi = np.clip(Phi, 1e-300, 1.0)
    return phi / Phi


class HeckmanSelection:
    r"""Heckman two-step sample selection correction model.

    Implements the Heckman (1979) two-step estimator and full MLE
    for sample selection bias correction.

    Parameters
    ----------
    max_iter : int, default=500
        Maximum Newton-Raphson iterations for probit.
    tol : float, default=1e-8
        Convergence tolerance.

    Attributes
    ----------
    gamma_ : ndarray of shape (q,)
        Probit coefficients (selection equation).
    beta_ : ndarray of shape (k,)
        OLS coefficients (outcome equation).
    sigma_ : float
        Residual standard deviation.
    rho_ : float
        Correlation between selection and outcome errors.
    lambda_ : ndarray of shape (n_selected,)
        Inverse Mills Ratio for selected observations.
    imr_coef_ : float
        Estimated coefficient on lambda (equals rho * sigma).
    """

    def __init__(self, max_iter=500, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.gamma_ = None
        self.beta_ = None
        self.sigma_ = None
        self.rho_ = None
        self.lambda_ = None
        self.imr_coef_ = None
        self.vcov_beta_ = None
        self.se_beta_ = None
        self.tstat_beta_ = None

    # ------------------------------------------------------------------
    # Probit via Newton-Raphson
    # ------------------------------------------------------------------
    def _probit(self, Z, D):
        r"""Estimate probit model P(D=1|Z) = Phi(Z*gamma).

        Maximizes the log-likelihood via Newton-Raphson:

        .. math::

            \\ell(\\gamma) = \\sum_{i} [D_i \\log\\Phi(Z_i\\gamma)
                            + (1-D_i) \\log(1 - \\Phi(Z_i\\gamma))]

            \\nabla\\ell = Z'(D - \\Phi)

            H = -Z' \\text{diag}(\\phi_i^2 / [\\Phi_i(1-\\Phi_i)]) Z

        Parameters
        ----------
        Z : ndarray of shape (n, q)
            Selection equation regressors.
        D : ndarray of shape (n,)
            Binary selection indicator.

        Returns
        -------
        gamma : ndarray of shape (q,)
            Probit coefficients.
        """
        Z = np.asarray(Z, dtype=float)
        D = np.asarray(D, dtype=float).ravel()
        n, q = Z.shape

        # Initialize (OLS on D as starting values, scaled)
        gamma = np.linalg.lstsq(Z, D, rcond=None)[0]

        for iteration in range(self.max_iter):
            z = Z @ gamma
            Phi = _norm_cdf(z)
            phi = _norm_pdf(z)

            # Log-likelihood
            Phi = np.clip(Phi, 1e-300, 1 - 1e-300)

            # Gradient
            g = Z.T @ (D - Phi)

            # Hessian (negative of information matrix)
            w = phi ** 2 / (Phi * (1.0 - Phi))
            H = Z.T @ (Z * w[:, np.newaxis])

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, g, rcond=None)[0]

            gamma_new = gamma + delta

            if np.max(np.abs(delta)) < self.tol:
                gamma = gamma_new
                break

            gamma = gamma_new

        return gamma

    # ------------------------------------------------------------------
    # Two-step estimator
    # ------------------------------------------------------------------
    def heckman_two_step(self, y, X, Z, D):
        r"""Heckman (1979) two-step sample selection estimator.

        **Step 1 — Probit selection:**

        .. math::

            P(D_i = 1 | Z_i) = \\Phi(Z_i\\gamma)

        **Step 2 — Augmented OLS:**

        .. math::

            y_i = X_i\\beta + \\beta_\\lambda \\cdot \\lambda(Z_i\\hat{\\gamma}) + \\eta_i

        where :math:`\\beta_\\lambda = \\rho\\sigma`.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Outcome variable (observed for selected).
        X : ndarray of shape (n, k)
            Outcome equation regressors.
        Z : ndarray of shape (n, q)
            Selection equation regressors.
        D : ndarray of shape (n,)
            Binary selection indicator (1 = observed).

        Returns
        -------
        dict
            Keys: 'gamma', 'beta', 'sigma', 'rho', 'imr_coef',
                  'lambda', 'll_probit', 'vcov_beta', 'se_beta', 'tstat_beta'.
        """
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()

        # Ensure consistent dimensions
        assert X.shape[0] == Z.shape[0] == len(y) == len(D)

        # ---- Step 1: Probit ----
        n = len(D)
        gamma = self._probit(Z, D)
        z_hat = Z @ gamma

        # Inverse Mills Ratio
        lam = _inverse_mills_ratio(z_hat)

        # Log-likelihood for probit
        Phi = np.clip(_norm_cdf(z_hat), 1e-300, 1 - 1e-300)
        ll_probit = np.sum(D * np.log(Phi) + (1.0 - D) * np.log(1.0 - Phi))

        # ---- Step 2: Augmented OLS (only for selected obs) ----
        sel = D == 1
        y_sel = y[sel]
        X_sel = X[sel]
        lam_sel = lam[sel]
        n_sel = np.sum(sel)

        # Augmented design matrix
        X_aug = np.column_stack([np.ones(X_sel.shape[0]), X_sel, lam_sel])
        k_aug = X_aug.shape[1]

        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y_sel

        beta_aug = np.linalg.solve(XtX, Xty)
        residuals = y_sel - X_aug @ beta_aug
        sigma2 = np.sum(residuals ** 2) / (n_sel - k_aug)
        sigma = np.sqrt(sigma2)

        # Extract components
        beta_const = beta_aug[0]
        beta_vec = beta_aug[1:1 + X.shape[1]]
        imr_coef = beta_aug[-1]

        # rho from imr_coef = rho * sigma
        rho = imr_coef / sigma if sigma > 1e-12 else 0.0
        rho = np.clip(rho, -0.999, 0.999)

        # ---- Heckman (1979) standard error correction ----
        # The two-step estimator uses a generated regressor (lambda),
        # so standard OLS SEs are wrong. Heckman's formula:
        #
        # Var(beta_hat) = sigma^2 (X*'X*)^(-1) [X*'X* - rho^2 X*' D X*](X*'X*)^(-1)
        #
        # plus additional terms involving the probit covariance.
        # Here we implement the simplified corrected covariance.

        # Compute delta_i = lambda_i * (lambda_i + Z_i*gamma)
        z_sel = z_hat[sel]
        delta = lam_sel * (lam_sel + z_sel)

        # D is diagonal matrix with delta_i
        # Correction for the generated regressor
        # Var(beta) = sigma^2 (X*'X*)^(-1) + sigma^2 (X*'X*)^(-1) X*' Delta X* (X*'X*)^(-1) * rho^2
        # where Delta = diag(delta_i)

        XtX_inv = np.linalg.inv(XtX)
        Delta = np.diag(delta)
        correction = rho ** 2 * XtX_inv @ X_aug.T @ Delta @ X_aug @ XtX_inv
        vcov_beta = sigma2 * XtX_inv + sigma2 * correction

        se_beta = np.sqrt(np.clip(np.diag(vcov_beta), 0, None))
        tstat_beta = np.where(se_beta > 1e-15, beta_aug / se_beta, 0.0)

        # Store attributes
        self.gamma_ = gamma
        self.beta_ = np.concatenate([[beta_const], beta_vec])
        self.sigma_ = sigma
        self.rho_ = rho
        self.imr_coef_ = imr_coef
        self.lambda_ = lam_sel
        self.vcov_beta_ = vcov_beta
        self.se_beta_ = se_beta
        self.tstat_beta_ = tstat_beta

        return {
            "gamma": gamma,
            "beta": self.beta_,
            "sigma": sigma,
            "rho": rho,
            "imr_coef": imr_coef,
            "lambda": lam_sel,
            "ll_probit": ll_probit,
            "vcov_beta": vcov_beta,
            "se_beta": se_beta,
            "tstat_beta": tstat_beta,
        }

    # ------------------------------------------------------------------
    # Full MLE
    # ------------------------------------------------------------------
    def heckman_mle(self, y, X, Z, D):
        r"""Full Maximum Likelihood Estimation of the Heckman selection model.

        The log-likelihood for the Type II Tobit (Heckman) model:

        .. math::

            \\ell = \\sum_{D_i=0} \\log\\Phi(-Z_i\\gamma) +
                    \\sum_{D_i=1} \\Big[\\log\\Phi\\Big(\\frac{Z_i\\gamma
                    + \\rho\\cdot\\frac{y_i - X_i\\beta}{\\sigma}}{\\sqrt{1-\\rho^2}}\\Big)
                    + \\log\\phi\\Big(\\frac{y_i - X_i\\beta}{\\sigma}\\Big) - \\log\\sigma\\Big]

        Parameters
        ----------
        y : ndarray of shape (n,)
            Outcome variable (arbitrary for unselected obs).
        X : ndarray of shape (n, k)
            Outcome equation regressors.
        Z : ndarray of shape (n, q)
            Selection equation regressors.
        D : ndarray of shape (n,)
            Binary selection indicator.

        Returns
        -------
        dict
            Estimated parameters: 'gamma', 'beta', 'sigma', 'rho', 'loglik'.
        """
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()

        # Initialize from two-step
        res_ts = self.heckman_two_step(y, X, Z, D)
        gamma0 = res_ts["gamma"]
        beta0 = res_ts["beta"]
        sigma0 = res_ts["sigma"]
        rho0 = res_ts["rho"]

        n, k = X.shape
        q = Z.shape[1]

        # Parameters to optimize: [gamma (q), beta_const, beta (k), log_sigma, atanh_rho]
        # We use atanh(rho) to keep rho in (-1, 1)
        atanh = lambda r: 0.5 * np.log((1.0 + np.clip(r, -0.999, 0.999)) /
                                        (1.0 - np.clip(r, -0.999, 0.999)))
        tanh = lambda a: np.tanh(a)

        log_sigma0 = np.log(max(sigma0, 1e-6))
        atanh_rho0 = atanh(rho0)

        theta0 = np.concatenate([
            gamma0,            # q params
            beta0[0:1],        # constant
            beta0[1:],         # k params (beta without constant)
            [log_sigma0],      # log(sigma)
            [atanh_rho0],      # atanh(rho)
        ])

        def _neg_loglik(theta):
            gamma_t = theta[:q]
            beta_c = theta[q]
            beta_t = theta[q + 1:q + 1 + k]
            log_sigma = theta[q + 1 + k]
            atanh_rho = theta[q + 1 + k + 1]

            sigma_t = np.exp(log_sigma)
            rho_t = tanh(atanh_rho)

            z = Z @ gamma_t
            resid = y - (beta_c + X @ beta_t)
            e = resid / sigma_t

            # Selected part
            sel = D == 1
            z_sel = z[sel]
            e_sel = e[sel]

            arg = (z_sel + rho_t * e_sel) / np.sqrt(max(1.0 - rho_t ** 2, 1e-12))
            Phi_cond = _norm_cdf(arg)
            Phi_cond = np.clip(Phi_cond, 1e-300, 1.0)

            ll_sel = np.sum(np.log(Phi_cond) - 0.5 * e_sel ** 2
                            - 0.5 * np.log(2.0 * np.pi) - log_sigma)

            # Not selected part
            nsel = D == 0
            z_nsel = z[nsel]
            phi_nsel = _norm_cdf(-z_nsel)
            phi_nsel = np.clip(phi_nsel, 1e-300, 1.0)
            ll_nsel = np.sum(np.log(phi_nsel))

            return -(ll_sel + ll_nsel)

        # Simple gradient-free optimization (Nelder-Mead style)
        # We'll use a modified Powell's method / coordinate descent
        best_theta = theta0.copy()
        best_f = _neg_loglik(best_theta)
        step_sizes = np.abs(best_theta) * 0.5 + 0.1

        for iteration in range(self.max_iter):
            improved = False
            for j in range(len(theta0)):
                # Evaluate at theta + step, theta - step
                theta_plus = best_theta.copy()
                theta_plus[j] += step_sizes[j]
                f_plus = _neg_loglik(theta_plus)

                theta_minus = best_theta.copy()
                theta_minus[j] -= step_sizes[j]
                f_minus = _neg_loglik(theta_minus)

                if f_plus < best_f and f_plus <= f_minus:
                    best_theta[j] = theta_plus[j]
                    best_f = f_plus
                    step_sizes[j] *= 1.2
                    improved = True
                elif f_minus < best_f:
                    best_theta[j] = theta_minus[j]
                    best_f = f_minus
                    step_sizes[j] *= 1.2
                    improved = True
                else:
                    step_sizes[j] *= 0.5

            if not improved or np.max(step_sizes) < self.tol:
                break

        # Extract results
        gamma_mle = best_theta[:q]
        beta_const_mle = best_theta[q]
        beta_mle = best_theta[q + 1:q + 1 + k]
        sigma_mle = np.exp(best_theta[q + 1 + k])
        rho_mle = tanh(best_theta[q + 1 + k + 1])

        return {
            "gamma": gamma_mle,
            "beta": np.concatenate([[beta_const_mle], beta_mle]),
            "sigma": sigma_mle,
            "rho": rho_mle,
            "loglik": -best_f,
        }

    # ------------------------------------------------------------------
    # Selection correction interpretation
    # ------------------------------------------------------------------
    def selection_correction(self, model):
        r"""Interpret the selection correction parameters.

        The coefficient on :math:`\\lambda` is :math:`\\rho\\sigma`.
        A significant :math:`\\rho` indicates selection bias.

        .. math::

            H_0: \\rho = 0 \\quad \\text{(no selection bias)}

        Parameters
        ----------
        model : dict
            Output from `heckman_two_step`.

        Returns
        -------
        dict
            Keys: 'rho', 'imr_coef', 'selection_bias_present' (bool),
                  'interpretation' (str).
        """
        rho = model["rho"]
        imr = model["imr_coef"]
        se_beta = model["se_beta"]
        tstat_beta = model["tstat_beta"]

        # The t-stat for the IMR coefficient (last entry in augmented regression)
        imr_tstat = tstat_beta[-1]
        imr_se = se_beta[-1]

        # Significance at 5% level (approximate critical value = 1.96)
        has_bias = np.abs(imr_tstat) > 1.96

        if has_bias and rho > 0:
            interp = (
                f"Significant positive selection bias (rho={rho:.4f}). "
                "Unobserved factors that increase the probability of selection "
                "also increase the outcome. OLS estimates would be biased upward."
            )
        elif has_bias and rho < 0:
            interp = (
                f"Significant negative selection bias (rho={rho:.4f}). "
                "Unobserved factors that increase the probability of selection "
                "decrease the outcome. OLS estimates would be biased downward."
            )
        else:
            interp = (
                f"No significant selection bias detected (rho={rho:.4f}, "
                f"t={imr_tstat:.3f}). Selection and outcome errors are not "
                "significantly correlated."
            )

        return {
            "rho": rho,
            "imr_coef": imr,
            "imr_se": imr_se,
            "imr_tstat": imr_tstat,
            "selection_bias_present": has_bias,
            "interpretation": interp,
        }
