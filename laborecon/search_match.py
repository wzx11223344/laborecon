"""
Diamond-Mortensen-Pissarides (DMP) Search and Matching Model
============================================================

Implements the canonical DMP labor market search model.

Core equations:
--------------

**Matching function** (Cobb-Douglas):

.. math::

    m(u, v) = \\mu \\cdot u^{\\eta} \\cdot v^{1-\\eta}

where :math:`u` = unemployment, :math:`v` = vacancies,
:math:`\\mu` = matching efficiency, :math:`\\eta` = elasticity w.r.t. unemployment.

**Labor market tightness**:

.. math::

    \\theta = \\frac{v}{u}, \\quad
    q(\\theta) = \\frac{m}{v} = \\mu \\theta^{-\\eta}

**Steady-state unemployment** (Beveridge curve):

.. math::

    u = \\frac{\\lambda}{\\lambda + \\theta q(\\theta)}

where :math:`\\lambda` = job separation rate.

**Job creation (free entry)**:

.. math::

    \\kappa = q(\\theta) \\cdot \\frac{y - w}{r + \\lambda}

**Nash bargaining**:

.. math::

    w = \\beta \\cdot y + (1 - \\beta) \\cdot b + \\beta \\cdot \\theta \\cdot \\kappa

where :math:`\\beta` = worker bargaining power, :math:`b` = unemployment benefit,
:math:`\\kappa` = vacancy posting cost, :math:`r` = discount rate.
"""

import numpy as np


class DiamondMortensenPissarides:
    r"""Diamond-Mortensen-Pissarides search and matching model.

    Solves the canonical equilibrium search model of the labor market.

    Parameters
    ----------
    params : dict, optional
        Default parameter values. Keys:
        - mu (float): matching efficiency, default 0.5
        - eta (float): matching elasticity w.r.t. u, default 0.5
        - lam (float): job separation rate, default 0.03
        - kappa (float): vacancy posting cost, default 0.2
        - beta (float): worker bargaining power, default 0.5
        - b (float): unemployment benefit / value of leisure, default 0.7
        - y (float): labor productivity, default 1.0
        - r (float): discount rate, default 0.01
    """

    def __init__(self, params=None):
        self.defaults = {
            "mu": 0.5,     # matching efficiency
            "eta": 0.5,    # matching elasticity w.r.t. unemployment
            "lam": 0.03,   # job separation rate
            "kappa": 0.2,  # vacancy posting cost
            "beta": 0.5,   # worker bargaining power
            "b": 0.7,      # unemployment benefit / flow value of non-employment
            "y": 1.0,      # labor productivity
            "r": 0.01,     # discount rate
        }
        if params:
            self.defaults.update(params)

    def _matching(self, u, v, mu, eta):
        r"""Matching function: :math:`m = \\mu \\cdot u^{\\eta} \\cdot v^{1-\\eta}`."""
        return mu * (u ** eta) * (v ** (1.0 - eta))

    def _job_finding_rate(self, theta, mu, eta):
        r"""Job finding rate: :math:`f(\\theta) = \\theta q(\\theta) = \\mu \\theta^{1-\\eta}`."""
        return mu * (theta ** (1.0 - eta))

    def _vacancy_filling_rate(self, theta, mu, eta):
        r"""Vacancy filling rate: :math:`q(\\theta) = \\mu \\theta^{-\\eta}`."""
        return mu * (theta ** (-eta))

    def steady_state(self, params=None):
        r"""Solve for the DMP steady-state equilibrium.

        The equilibrium is defined by three equations:

        **Job creation (JC):**

        .. math::

            \\kappa = q(\\theta) \\cdot \\frac{y - w}{r + \\lambda}

        **Wage curve (W):**

        .. math::

            w = \\beta \\cdot y + (1 - \\beta) \\cdot b + \\beta \\cdot \\theta \\cdot \\kappa

        **Beveridge curve (BC):**

        .. math::

            u = \\frac{\\lambda}{\\lambda + f(\\theta)}

        We solve for :math:`\\theta` from the JC-Wage intersection:

        .. math::

            w = y - \\frac{\\kappa(r + \\lambda)}{q(\\theta)}

            w = \\beta y + (1 - \\beta)b + \\beta\\theta\\kappa

        Equating and solving:

        .. math::

            \\frac{\\kappa(r + \\lambda)}{q(\\theta)} + \\beta\\theta\\kappa
            = (1 - \\beta)(y - b)

        Parameters
        ----------
        params : dict, optional
            Model parameters (overrides defaults).

        Returns
        -------
        dict
            Keys: 'theta' (tightness), 'u' (unemployment rate),
                  'v' (vacancy rate), 'w' (wage), 'f' (job finding rate),
                  'q' (vacancy filling rate), 'params'.
        """
        p = dict(self.defaults)
        if params:
            p.update(params)

        mu, eta, lam, kappa, beta, b, y, r = (
            p["mu"], p["eta"], p["lam"], p["kappa"],
            p["beta"], p["b"], p["y"], p["r"]
        )

        # Solve for theta from:
        # kappa*(r+lam)/q(theta) + beta*theta*kappa = (1-beta)*(y-b)
        # => kappa*(r+lam)/(mu*theta^(-eta)) + beta*theta*kappa = (1-beta)*(y-b)
        # => kappa*(r+lam)*theta^eta/mu + beta*kappa*theta = (1-beta)*(y-b)
        #
        # F(theta) = kappa*(r+lam)*theta^eta/mu + beta*kappa*theta - (1-beta)*(y-b) = 0

        rhs = (1.0 - beta) * (y - b)

        def f_theta(th):
            return kappa * (r + lam) * (th ** eta) / mu + beta * kappa * th - rhs

        # Bisection search for theta
        theta_low, theta_high = 0.001, 50.0
        for _ in range(200):
            theta_mid = 0.5 * (theta_low + theta_high)
            f_mid = f_theta(theta_mid)
            if f_mid > 0:
                theta_high = theta_mid
            else:
                theta_low = theta_mid
            if theta_high - theta_low < 1e-12:
                break

        theta = 0.5 * (theta_low + theta_high)

        # Job finding and vacancy filling rates
        f_rate = self._job_finding_rate(theta, mu, eta)
        q_rate = self._vacancy_filling_rate(theta, mu, eta)

        # Steady-state unemployment (Beveridge curve)
        u = lam / (lam + f_rate)

        # Vacancy rate
        v = theta * u

        # Wage from Nash bargaining or JC curve
        w = beta * y + (1.0 - beta) * b + beta * theta * kappa

        # Verify with JC: w_jc = y - kappa*(r+lam)/q
        w_jc = y - kappa * (r + lam) / q_rate
        # Use average for robustness
        w = 0.5 * (w + w_jc)

        return {
            "theta": theta,
            "u": u,
            "v": v,
            "w": w,
            "f": f_rate,
            "q": q_rate,
            "params": p,
        }

    def comparative_statics(self, param_range, vary_param="y", params=None):
        r"""Comparative statics: vary a parameter over a range and compute
        equilibrium outcomes.

        Parameters
        ----------
        param_range : ndarray
            Values of the parameter to vary.
        vary_param : str, default='y'
            Parameter name to vary ('y', 'b', 'kappa', 'lam', 'beta', 'mu', 'eta', 'r').
        params : dict, optional
            Base parameters.

        Returns
        -------
        dict
            Keys: 'param_values', 'theta', 'u', 'w', 'v', 'f', 'vary_param'.
        """
        p = dict(self.defaults)
        if params:
            p.update(params)

        results = {"theta": [], "u": [], "w": [], "v": [], "f": []}

        for val in param_range:
            p[vary_param] = val
            ss = self.steady_state(p)
            results["theta"].append(ss["theta"])
            results["u"].append(ss["u"])
            results["w"].append(ss["w"])
            results["v"].append(ss["v"])
            results["f"].append(ss["f"])

        results["param_values"] = np.array(param_range)
        results["vary_param"] = vary_param

        for key in ["theta", "u", "w", "v", "f"]:
            results[key] = np.array(results[key])

        return results

    def beveridge_curve(self, params=None, v_range=None):
        r"""Compute the Beveridge curve: the unemployment-vacancy locus.

        The Beveridge curve shows the inverse relationship between
        unemployment :math:`u` and vacancies :math:`v` in steady state:

        .. math::

            u = \\frac{\\lambda}{\\lambda + \\mu \\theta^{1-\\eta}},
            \\quad \\theta = v/u

        Parameters
        ----------
        params : dict, optional
            Model parameters.
        v_range : ndarray, optional
            Range of vacancy rates. Default: linspace(0.005, 0.2, 200).

        Returns
        -------
        dict
            Keys: 'u' (unemployment), 'v' (vacancies), 'theta' (tightness).
        """
        p = dict(self.defaults)
        if params:
            p.update(params)

        mu, eta, lam = p["mu"], p["eta"], p["lam"]

        if v_range is None:
            v_range = np.linspace(0.005, 0.2, 200)

        v_arr = np.asarray(v_range)
        theta_vals = np.zeros_like(v_arr)
        u_vals = np.zeros_like(v_arr)

        for i, v in enumerate(v_arr):
            # Solve: u such that m(u, v) = lam*(1-u)
            # mu * u^eta * v^(1-eta) = lam * (1-u)
            #
            # u is small, so use bisection
            u_low, u_high = 0.001, 0.3
            for _ in range(100):
                u_mid = 0.5 * (u_low + u_high)
                inflow = lam * (1.0 - u_mid)
                outflow = mu * (u_mid ** eta) * (v ** (1.0 - eta))
                if outflow > inflow:
                    u_high = u_mid
                else:
                    u_low = u_mid
            u_vals[i] = 0.5 * (u_low + u_high)
            theta_vals[i] = v / u_vals[i]

        return {"u": u_vals, "v": v_arr, "theta": theta_vals}

    def calibrate(self, targets):
        r"""Calibrate model parameters to match observed targets.

        Calibrates matching efficiency :math:`\\mu` and bargaining power
        :math:`\\beta` to match observed unemployment rate and labor
        market tightness.

        Parameters
        ----------
        targets : dict
            Keys:
            - 'u' (float): target unemployment rate.
            - 'theta' (float): target labor market tightness.
            Additional optional keys: 'w', 'v'.

        Returns
        -------
        dict
            Calibrated parameters (including 'mu' and 'beta').
        """
        p = dict(self.defaults)

        u_target = targets["u"]
        theta_target = targets["theta"]

        # From steady-state: u = lam/(lam + f(theta))
        # => f = lam*(1-u)/u
        # => mu * theta^(1-eta) = lam*(1-u)/u
        # => mu = lam*(1-u)/(u * theta^(1-eta))

        lam, eta = p["lam"], p["eta"]
        mu_cal = lam * (1.0 - u_target) / (u_target * (theta_target ** (1.0 - eta)))
        p["mu"] = mu_cal

        # From JC + Wage: kappa*(r+lam)/q + beta*theta*kappa = (1-beta)*(y-b)
        # => kappa*(r+lam)/(mu*theta^(-eta)) + beta*theta*kappa = (1-beta)*(y-b)
        # => kappa*(r+lam)*theta^eta/mu + beta*theta*kappa = (1-beta)*(y-b)
        # => beta*(theta*kappa + (y-b)) = (y-b) - kappa*(r+lam)*theta^eta/mu
        # => beta = [(y-b) - kappa*(r+lam)*theta^eta/mu] / [theta*kappa + (y-b)]

        kappa, b, y, r = p["kappa"], p["b"], p["y"], p["r"]
        numerator = (y - b) - kappa * (r + lam) * (theta_target ** eta) / mu_cal
        denominator = theta_target * kappa + (y - b)
        beta_cal = numerator / denominator
        beta_cal = np.clip(beta_cal, 0.01, 0.99)
        p["beta"] = beta_cal

        # Recompute steady state
        ss = self.steady_state(p)

        return {
            "params_calibrated": p,
            "mu_calibrated": mu_cal,
            "beta_calibrated": beta_cal,
            "steady_state": ss,
            "u_match": np.abs(ss["u"] - u_target),
            "theta_match": np.abs(ss["theta"] - theta_target),
        }
