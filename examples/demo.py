"""
LaborEcon Demo
==============

Demonstrates key functionality across all modules in the Labor Economics Toolkit.
"""

import numpy as np

# ============================================================
# 1. Mincer Earnings Equation
# ============================================================
print("=" * 70)
print("1. Mincer Earnings Equation")
print("=" * 70)

from laborecon.mincer import MincerEquation

# Generate synthetic data
np.random.seed(42)
n = 1000

edu = np.random.normal(12, 3, n)        # years of education
exp = np.random.uniform(0, 45, n)        # potential experience
exp_sq = exp ** 2

# True parameters: return to edu = 8%, exp profile concave
log_wage = (0.5 + 0.08 * edu + 0.04 * exp - 0.0006 * exp_sq
            + np.random.normal(0, 0.3, n))

X = np.column_stack([edu, exp, exp_sq])
y = log_wage

mincer = MincerEquation(add_constant=True)
results = mincer.fit(X, y)

print(f"\nMincer OLS Results:")
print(f"  R-squared = {results.r2_:.4f}")
print(f"  Adj R-squared = {results.adj_r2_:.4f}")

# Returns to education
ret = mincer.returns_to_education(results)
print(f"\nReturns to Education:")
print(f"  Return = {ret['return_pct']:.2f}% per year")
print(f"  t-statistic = {ret['tstat']:.2f}")
print(f"  p-value = {ret['pval']:.4f}")

# Experience profile
profile = mincer.experience_profile(results, years=40)
peak_idx = np.argmax(profile["predicted_wage"])
print(f"\nExperience Profile:")
print(f"  Peak wage at {profile['experience'][peak_idx]:.0f} years of experience")
print(f"  Peak predicted wage = {profile['predicted_wage'][peak_idx]:.2f}")

# ============================================================
# 2. Heckman Selection Model
# ============================================================
print("\n" + "=" * 70)
print("2. Heckman Selection Model")
print("=" * 70)

from laborecon.heckman import HeckmanSelection

np.random.seed(123)
n_heck = 2000

# Generate true process with selection
X_heck = np.column_stack([
    np.random.normal(12, 3, n_heck),  # education
    np.random.uniform(0, 40, n_heck), # experience
])

# Selection equation
Z_heck = np.column_stack([
    np.ones(n_heck),
    X_heck[:, 1] / 40,              # experience (scaled)
    np.random.normal(0, 1, n_heck),  # exclusion restriction (instrument)
])

gamma_true = np.array([0.5, -0.8, 1.2])
rho_true = 0.6
sigma_true = 0.5

u = np.random.normal(0, 1, n_heck)
eps = rho_true * sigma_true * u + np.sqrt(1 - rho_true ** 2) * sigma_true * np.random.normal(0, 1, n_heck)

D_star = Z_heck @ gamma_true + u
D = (D_star > 0).astype(float)

beta_true = np.array([0.3, 0.07, 0.03])
y_heck = beta_true[0] + X_heck @ beta_true[1:] + eps

print(f"\nSelection rate: {np.mean(D):.2%}")
print(f"True rho = {rho_true:.3f}, sigma = {sigma_true:.3f}")

# Two-step estimation
heckman = HeckmanSelection(max_iter=500, tol=1e-8)
res_ts = heckman.heckman_two_step(y_heck, X_heck, Z_heck, D)

print(f"\nTwo-Step Results:")
print(f"  rho = {res_ts['rho']:.4f} (true: {rho_true})")
print(f"  sigma = {res_ts['sigma']:.4f} (true: {sigma_true})")
print(f"  IMR coefficient = {res_ts['imr_coef']:.4f}")

# Selection correction interpretation
corr = heckman.selection_correction(res_ts)
print(f"\nSelection Correction:")
print(f"  {corr['interpretation']}")

# Full MLE
res_mle = heckman.heckman_mle(y_heck, X_heck, Z_heck, D)
print(f"\nMLE Results:")
print(f"  rho = {res_mle['rho']:.4f} (true: {rho_true})")
print(f"  sigma = {res_mle['sigma']:.4f} (true: {sigma_true})")

# Compare with naive OLS (ignoring selection)
from laborecon.mincer import MincerEquation
naive = MincerEquation(add_constant=True)
naive.fit(X_heck[D == 1], y_heck[D == 1])
print(f"\nNaive OLS (ignoring selection):")
print(f"  Education coefficient = {naive.beta_[1]:.4f} (true: {beta_true[1]:.3f})")

# ============================================================
# 3. Oaxaca-Blinder Decomposition
# ============================================================
print("\n" + "=" * 70)
print("3. Oaxaca-Blinder Decomposition")
print("=" * 70)

from laborecon.decomposition import OaxacaBlinder

np.random.seed(456)
n_ob = 500

# Group A (e.g., males) and Group B (e.g., females)
X_A = np.column_stack([
    np.random.normal(14, 2.5, n_ob),
    np.random.uniform(5, 35, n_ob),
])

X_B = np.column_stack([
    np.random.normal(13, 2.8, n_ob),  # slightly less education
    np.random.uniform(3, 30, n_ob),   # slightly less experience
])

beta_true_ob = np.array([0.5, 0.08, 0.03])
y_A = beta_true_ob[0] + X_A @ beta_true_ob[1:] + np.random.normal(0, 0.4, n_ob)
y_B = beta_true_ob[0] + X_B @ beta_true_ob[1:] + np.random.normal(0, 0.4, n_ob)

ob = OaxacaBlinder(add_constant=True)
decomp = ob.oaxaca_blinder(y_A, X_A, y_B, X_B)

print(f"\nRaw wage gap (A - B): {decomp['raw_gap']:.4f}")
print(f"  Endowment effect  : {decomp['endowment']:.4f} ({decomp['endowment']/decomp['raw_gap']*100:.1f}%)")
print(f"  Coefficient effect: {decomp['coefficient']:.4f} ({decomp['coefficient']/decomp['raw_gap']*100:.1f}%)")
print(f"  Interaction       : {decomp['interaction']:.4f} ({decomp['interaction']/decomp['raw_gap']*100:.1f}%)")
print(f"  Check sum = {decomp['check_sum']:.6f} (should equal raw_gap)")

# Detailed decomposition
detail = ob.detailed_decomposition(decomp, var_names=["const", "Education", "Experience"])
print(f"\nDetailed Decomposition:")
print(f"  Endowment by variable:")
for k, v in detail["endowment"].items():
    print(f"    {k}: {v:.4f}")

# Juhn-Murphy-Pierce
jmp = ob.juhn_murphy_pierce(y_A, X_A, y_B, X_B, n_quantiles=50)
print(f"\nJMP Decomposition (median):")
mid = len(jmp["quantiles"]) // 2
print(f"  Gap at median = {jmp['gap'][mid]:.4f}")
print(f"  Quantity/Price effect = {jmp['quantity_price_effect'][mid]:.4f}")
print(f"  Residual effect = {jmp['residual_effect'][mid]:.4f}")

# ============================================================
# 4. DMP Search and Matching Model
# ============================================================
print("\n" + "=" * 70)
print("4. DMP Search and Matching Model")
print("=" * 70)

from laborecon.search_match import DiamondMortensenPissarides

dmp = DiamondMortensenPissarides()

# Steady state
ss = dmp.steady_state()
print(f"\nSteady State Equilibrium:")
print(f"  Labor market tightness (theta) = {ss['theta']:.4f}")
print(f"  Unemployment rate (u)         = {ss['u']:.4f}")
print(f"  Vacancy rate (v)              = {ss['v']:.4f}")
print(f"  Wage (w)                      = {ss['w']:.4f}")
print(f"  Job finding rate (f)          = {ss['f']:.4f}")
print(f"  Vacancy filling rate (q)      = {ss['q']:.4f}")

# Comparative statics: vary productivity y
y_range = np.linspace(0.5, 1.5, 50)
cs = dmp.comparative_statics(y_range, vary_param="y")

print(f"\nComparative Statics (varying productivity y from {y_range[0]} to {y_range[-1]}):")
print(f"  u range: [{cs['u'][0]:.4f}, {cs['u'][-1]:.4f}]")
print(f"  w range: [{cs['w'][0]:.4f}, {cs['w'][-1]:.4f}]")
print(f"  theta range: [{cs['theta'][0]:.4f}, {cs['theta'][-1]:.4f}]")

# Comparative statics: vary UI benefit b
b_range = np.linspace(0.3, 0.9, 50)
cs_b = dmp.comparative_statics(b_range, vary_param="b")
print(f"\nComparative Statics (varying UI benefit b from {b_range[0]} to {b_range[-1]}):")
print(f"  u range: [{cs_b['u'][0]:.4f}, {cs_b['u'][-1]:.4f}]")
print(f"  w range: [{cs_b['w'][0]:.4f}, {cs_b['w'][-1]:.4f}]")

# Beveridge curve
bc = dmp.beveridge_curve()
print(f"\nBeveridge Curve (u vs v):")
print(f"  u range: [{bc['u'][0]:.4f}, {bc['u'][-1]:.4f}]")
print(f"  v range: [{bc['v'][0]:.4f}, {bc['v'][-1]:.4f}]")

# Calibration
cal = dmp.calibrate({"u": 0.06, "theta": 0.72})
print(f"\nCalibration to u=0.06, theta=0.72:")
print(f"  Calibrated mu   = {cal['mu_calibrated']:.4f}")
print(f"  Calibrated beta = {cal['beta_calibrated']:.4f}")
print(f"  u match error   = {cal['u_match']:.6f}")
print(f"  theta match error = {cal['theta_match']:.6f}")

# ============================================================
# 5. Quantile Regression
# ============================================================
print("\n" + "=" * 70)
print("5. Quantile Regression & Wage Decomposition")
print("=" * 70)

from laborecon.quantile import QuantileRegression

# Generate heteroskedastic data
np.random.seed(789)
n_qr = 200
X_qr = np.random.normal(10, 3, n_qr)
# Heteroskedastic errors (larger variance at higher X)
eps_qr = np.random.normal(0, 1, n_qr) * (0.1 + 0.1 * X_qr)
y_qr = 2.0 + 0.8 * X_qr + eps_qr

qr = QuantileRegression(add_constant=True, max_iter=3000)
qr.fit(X_qr.reshape(-1, 1), y_qr, tau=0.5)
print(f"\nMedian Regression (tau=0.5):")
print(f"  Intercept = {qr.beta_[0]:.4f}")
print(f"  Slope     = {qr.beta_[1]:.4f}")

# Quantile wage decomposition
n_qd = 300
X_qd_A = np.column_stack([
    np.random.normal(14, 2.5, n_qd),
    np.random.uniform(5, 35, n_qd),
])
X_qd_B = np.column_stack([
    np.random.normal(12, 2.8, n_qd),
    np.random.uniform(3, 30, n_qd),
])

# Group B has lower returns
beta_qd_A = np.array([0.5, 0.10, 0.04])
beta_qd_B = np.array([0.5, 0.07, 0.03])

X_qd_A_aug = np.column_stack([np.ones(n_qd), X_qd_A])
X_qd_B_aug = np.column_stack([np.ones(n_qd), X_qd_B])
y_qd_A = X_qd_A_aug @ beta_qd_A + np.random.normal(0, 0.3, n_qd)
y_qd_B = X_qd_B_aug @ beta_qd_B + np.random.normal(0, 0.3, n_qd)

# Use a fast version (smaller max_iter for the demo)
qr_dec = QuantileRegression(add_constant=True, max_iter=2000)
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
qd_results = qr_dec.quantile_wage_decomposition(y_qd_A, X_qd_A, y_qd_B, X_qd_B,
                                                  quantiles=quantiles)

print(f"\nQuantile Wage Decomposition:")
print(f"  {'Quantile':>10} {'Total Gap':>10} {'Coef Effect':>12} {'Char Effect':>12}")
for i, qq in enumerate(quantiles):
    print(f"  {qq:10.2f} {qd_results['total_gap'][i]:10.4f} "
          f"{qd_results['coefficient_effect'][i]:12.4f} "
          f"{qd_results['characteristics_effect'][i]:12.4f}")

# Wage inequality trends (simulated panel)
print(f"\nWage Inequality Trends (simulated):")
np.random.seed(999)
trend_data = {}
for yr in range(2000, 2010):
    # Expanding inequality over time
    sigma_t = 0.3 + 0.03 * (yr - 2000)
    trend_data[yr] = np.random.normal(3.0, sigma_t, 500)

trends = qr_dec.wage_inequality_trends(trend_data, list(range(2000, 2010)))
print(f"  {'Year':>6} {'P90-P10':>8} {'P90-P50':>8} {'P50-P10':>8}")
for i, yr in enumerate(trends["years"]):
    print(f"  {int(yr):6d} {trends['p90_p10'][i]:8.4f} "
          f"{trends['p90_p50'][i]:8.4f} {trends['p50_p10'][i]:8.4f}")

print("\n" + "=" * 70)
print("Demo complete.")
print("=" * 70)
