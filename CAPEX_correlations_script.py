import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Data
x = np.array([])   # size
y = np.array([])  # cost €/kW

# Helper: power-law model
def model(x, a, k):
    return a * x**(-k)


# 1) Log-log linear regression (fast, gives initial guess)
log_x = np.log(x)
log_y = np.log(y)
slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
k_lin = -slope
a_lin = np.exp(intercept)
r2_log_lin = r_value**2

print("=== Log-log linear regression (on ln) ===")
print(f"linregress fit: y = {a_lin:.3f} * x^(-{k_lin:.3f})")
print(f"R^2 (log-space) = {r2_log_lin:.4f}")
print()


# 2) Nonlinear fit with curve_fit (directly in linear space)
#    uses linregress results as initial guess (p0)
p0 = [a_lin, k_lin]  # initial guess
params, cov = curve_fit(model, x, y, p0=p0, maxfev=10000)
a_cf, k_cf = params
# parameter uncertainties (1-sigma)
perr = np.sqrt(np.diag(cov))

print("=== Nonlinear regression (curve_fit) ===")
print(f"curve_fit: y = {a_cf:.3f} * x^(-{k_cf:.3f})")
print(f"parameter uncertainties (1-sigma): a = {perr[0]:.3f}, k = {perr[1]:.3f}")


# Goodness-of-fit: linear-space (on y) and log-space (on ln(y))
# predictions at the original x
y_pred_linreg = a_lin * x**(-k_lin)  # using linregress-derived params
y_pred_cf     = a_cf  * x**(-k_cf)   # using curve_fit params

# linear-space R^2
ss_res_cf = np.sum((y - y_pred_cf)**2)
ss_tot    = np.sum((y - np.mean(y))**2)
r2_linear_cf = 1 - ss_res_cf / ss_tot

# log-space R^2 (on ln(y)), for direct comparison with linregress R^2
ln_y_pred_cf = np.log(y_pred_cf)
ss_res_log_cf = np.sum((log_y - ln_y_pred_cf)**2)
ss_tot_log = np.sum((log_y - np.mean(log_y))**2)
r2_log_cf = 1 - ss_res_log_cf / ss_tot_log

print()
print("Goodness-of-fit (curve_fit):")
print(f"R^2 (linear-space) = {r2_linear_cf:.4f}")
print(f"R^2 (log-space)    = {r2_log_cf:.4f}")
print()

# Smooth x for plotting fitted curves
x_fit = np.linspace(min(x), max(x), 300)
y_fit_linreg = a_lin * x_fit**(-k_lin)
y_fit_cf     = a_cf  * x_fit**(-k_cf)


# Plot 1: Log–Log plot (diagnostic)
plt.figure(figsize=(7,5))
plt.scatter(x, y, label='Data', marker='o')
plt.plot(x_fit, y_fit_linreg,   label=f'ln fit: y={a_lin:.1f}x^-{k_lin:.2f}', linestyle='--')
plt.plot(x_fit, y_fit_cf,       label=f'curve_fit: y={a_cf:.1f}x^-{k_cf:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Size (kW)')
plt.ylabel('Cost (€/kW)')
plt.title('Log–Log Plot of Cost vs Size')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()


# Plot 2: Normal (linear) axes
plt.figure(figsize=(7,5))
plt.scatter(x, y, label='Data', marker='o')
plt.plot(x_fit, y_fit_linreg,   label=f'ln fit: y={a_lin:.1f}x^-{k_lin:.2f}', linestyle='--')
plt.plot(x_fit, y_fit_cf,       label=f'curve_fit: y={a_cf:.1f}x^-{k_cf:.2f}')
plt.xlabel('Size (kW)')
plt.ylabel('Cost (€/kW)')
plt.title('Cost vs Size (Normal Axes)')
plt.legend()
plt.grid(True, ls="--", lw=0.5)
plt.tight_layout()
plt.show()
