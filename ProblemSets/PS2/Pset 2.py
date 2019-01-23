#%%
# MACS 402 - Structural Analysis - Dr. Evans - Problem Set 2
# DATE: January 23rd, 2019
# AUTHOR: Adam Oppenheimer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.stats.distributions import chi2
import math
from math import e
#%%
#Problem 1
data = np.loadtxt("/Volumes/GoogleDrive/My Drive/3rd Year/Quarter 2" +\
                  "/Structural Estimation/StructEst_W19/ProblemSets/" +\
                  "PS2/clms.txt", dtype=np.float128)
#%%
#Problem 1 Part a
#Problem 1 Part a - Define Functions
def plot_graph(data, num_bins, weights, title, x_label, y_label):
        plt.hist(data, bins=num_bins, weights=weights)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
#%%
#Problem 1 Part a - Define Values
count = data.shape[0]
mean = data.mean()
median = np.median(data)
max_min = (data.max(), data.min())
std = data.std()
var = data.var()
print("Mean: ", mean)
print("Median: ", median)
print("Max: ", max_min[0])
print("Min: ", max_min[1])
print("Std: ", std)
#%%
#Problem 1 Part a - Graph 1
title = "Histogram of Health Claims"
x_label = "Value of Monthly Health Expenditures ($)"
y_label = "Percent of Observations"
plot_1_weights = (1 / count) * np.ones_like(data)
num_bins = 1000
plot_graph(data, num_bins, plot_1_weights, title, x_label, y_label)
plt.show()
#%%
#Problem 1 Part a - Graph 2
title = "Histogram of Health Claims <= $800"
data_under_800 = data[data <= 800]
plot_2_weights = (1 / count) * np.ones_like(data_under_800)
num_bins = 100
plot_graph(data_under_800, num_bins, plot_2_weights, title, x_label, y_label)
plt.show()
#%%
#Problem 1 Part b
#Problem 1 Part b - Define Functions
def gamma_fun_pdf(xvals, alpha, beta):
        pdf_vals = ( xvals ** (alpha - 1) * e ** ( - xvals / beta ) )/\
                   ( beta ** alpha * math.gamma(alpha) )
        return pdf_vals

def log_lik_fun_b(xvals, alpha, beta):
        pdf_vals = gamma_fun_pdf(xvals, alpha, beta)
        ln_pdf_vals = np.log(pdf_vals)
        return ln_pdf_vals.sum()

def crit_b(params, *args):
        alpha, beta = params
        xvals, = args
        log_lik_val = log_lik_fun_b(xvals, alpha, beta)
        return -log_lik_val
#%%
#Problem 1 Part b - Define Values
beta_0 = var/mean
alpha_0 = mean/beta_0
params_init = np.array([alpha_0, beta_0])
#%%
#Problem 1 Part b - Call Functions
log_lik = log_lik_fun_b(data, alpha_0, beta_0)
results_cstr = opt.minimize(crit_b, params_init,\
                 args=(data), method="L-BFGS-B",\
                 bounds=((1e-10, None), (1e-10, None)))
alpha_MLE_b, beta_MLE_b = results_cstr.x
log_lik_b = log_lik_fun_b(data, alpha_MLE_b, beta_MLE_b)
#%%
#Problem 1 Part b - Output Results
print("alpha_MLE_b=", alpha_MLE_b, " beta_MLE_b=", beta_MLE_b)
print("Maximize Log Likelihood: ", log_lik_b)
#%%
#Problem 1 Part b - Print Graphs
title = "Histogram of Health Claims <= $800 + (b) Prediction"
plot_graph(data_under_800, num_bins, plot_2_weights, title, x_label, y_label)
dist_pts = np.linspace(1e-10, data_under_800.max(), count)
plt.plot(dist_pts, gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b),\
        linewidth=2, label="Gamma", color="r")
plt.legend(loc="upper left")
plt.ylim([0, 0.05])
plt.xlim([0, 800])
plt.show()
#%%
#Problem 1 Part c
#Problem 1 Part c - Define Functions
def gen_gamma_fun_pdf(xvals, alpha, beta, m):
        pdf_vals = (m * e ** ( - ( xvals / beta ) ** m ))/\
                   (xvals * math.gamma(alpha / m)) *\
                   (xvals / beta) ** alpha
        return pdf_vals

def log_sum_c(xvals, alpha, beta, m):
        log_vals = np.log(m) + (alpha - 1) * np.log(xvals) -\
                   (xvals / beta) ** m - alpha * np.log(beta) -\
                   np.log(math.gamma(alpha / m))
        return log_vals.sum()

def crit_c(params, *args):
        alpha, beta, m = params
        xvals, = args
        log_lik_val = log_sum_c(xvals, alpha, beta, m)
        return -log_lik_val
#%%
#Problem 1 Part c - Define Values
alpha_0 = alpha_MLE_b
beta_0 = beta_MLE_b
m_0 = 1
params_init = np.array([alpha_0, beta_0, m_0])
#%%
#Problem 1 Part c - Call Functions
results_cstr = opt.minimize(crit_c, params_init,\
                              args=(data), method="L-BFGS-B",\
                              bounds=((1e-10, None), (1e-10, None),\
                              (1e-10, None)))
alpha_MLE_c, beta_MLE_c, m_MLE_c = results_cstr.x
log_lik_c = log_sum_c(data, alpha_MLE_c, beta_MLE_c, m_MLE_c)
#%%
#Problem 1 Part c - Output Results
print("alpha_MLE_c=", alpha_MLE_c, " beta_MLE_c=", beta_MLE_c, " m_MLE_c=", m_MLE_c)
print("Maximize Log Likelihood: ", log_lik_c)
#%%
#Problem 1 Part c - Print Graphs
title = "Histogram of Health Claims <= $800 + (c) Prediction"
plot_graph(data_under_800, num_bins, plot_2_weights, title, x_label, y_label)
plt.plot(dist_pts, gen_gamma_fun_pdf(dist_pts, alpha_MLE_c, beta_MLE_c, m_MLE_c),\
        linewidth=2, label="Generalized Gamma", color="r")
plt.legend(loc="upper left")
plt.ylim([0, 0.05])
plt.xlim([0, 800])
plt.show()
#%%
#Problem 1 Part d
#Problem 1 Part d - Define Functions
def gen_beta2_fun_pdf(xvals, a, b, p, q):
        pdf_vals = (xvals / b) ** (a * p) * a / (xvals * gb2_beta(p, q) *\
                    (1 + (xvals / b) ** a) ** (p + q) )
        return pdf_vals

def log_sum_d(xvals, a, b, p, q):
        log_vals = (a * p - 1) * np.log(xvals) + np.log(a) -\
                   a * p * np.log(b) - np.log(gb2_beta(p, q))\
                   - (p + q) * np.log(1 + (xvals / b) ** a)
        return log_vals.sum()

def gb2_beta(p, q):
        betainc = scipy.special.betainc(p, q, 1)
        return betainc * scipy.special.beta(p, q)

def crit_d(params, *args):
        a, b, p, q = params
        xvals, = args
        log_lik_val = log_sum_d(xvals, a, b, p, q)
        return -log_lik_val
#%%
#Problem 1 Part d - Define Values
a_0 = alpha_MLE_c
b_0 = beta_MLE_c
p_0 = m_MLE_c
q_0 = 10000
params_init = np.array([a_0, b_0, p_0, q_0])
#%%
#Problem 1 Part d - Call Functions
results_cstr = opt.minimize(crit_d, params_init,\
                            args=(data), method="L-BFGS-B",\
                            bounds=((1e-10, None), (1e-10, None),\
                            (1e-10, None), (1e-10, None)))
a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d = results_cstr.x
log_lik_d = log_sum_d(data, a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d)
#%%
#Problem 1 Part d - Output Results
print("a_MLE_d=", a_MLE_d, " b_MLE_d=", b_MLE_d,\
      " p_MLE_d=", p_MLE_d, " q_MLE_d=", q_MLE_d)
print("Maximize Log Likelihood: ", log_lik_d)
#%%
#Problem 1 Part d - Print Graphs
title = "Histogram of Health Claims <= $800 + (d) Prediction"
plot_graph(data_under_800, num_bins, plot_2_weights, title, x_label, y_label)
plt.plot(dist_pts, gen_beta2_fun_pdf(dist_pts, a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d),\
        linewidth=2, label="Generalized Beta 2", color="r")
plt.legend(loc="upper left")
plt.ylim([0, 0.05])
plt.xlim([0, 800])
plt.show()
#%%
#Problem 1 Part e
#Problem 1 Part e - Define Functions
def likelihood_ratio(log_lik_a, log_lik_b):
        return float(2 * (log_lik_a - log_lik_b))
#%%
#Problem 1 Part e - Call Functions
lik_rat_b_d = likelihood_ratio(log_lik_d, log_lik_b)
p_val_b_d = chi2.sf(lik_rat_b_d, 2)
lik_rat_c_d = likelihood_ratio(log_lik_d, log_lik_c)
p_val_c_d = chi2.sf(lik_rat_c_d, 1)
#%%
#Problem 1 Part e - Output Results
print("Likelihood ratio b-d: ", lik_rat_b_d)
print("P val b-d: ", p_val_b_d)
print("Likelihood ratio c-d: ", lik_rat_c_d)
print("P val c-d: ", p_val_c_d)
#%%
#Problem 1 Part f
#Problem 1 Part f - Define Functions
def integrand_d(x, a, b, p, q):
        return gen_beta2_fun_pdf(x, a, b, p, q)
def integrand_b(x, alpha, beta):
        return gamma_fun_pdf(x, alpha, beta)
#%%
#Problem 1 Part f - Call Functions
zero_to_1k_d = integrate.quad(integrand_d, 1e-10, 1000,\
                              args=(a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d))[0]
zero_to_1k_b = integrate.quad(integrand_b, 1e-10, 1000,\
                              args=(alpha_MLE_b, beta_MLE_b))[0]
#%%
#Problem 1 Part f - Output Results
print("Prob of >= $1,000 from b: ", 1-zero_to_1k_b)
print("Prob of >= $1,000 from d: ", 1-zero_to_1k_d)
print("Difference between b and d: ", abs(zero_to_1k_d - zero_to_1k_b))
#%%
#Problem 1 Final Graph:
title = "Histogram of Health Claims <= $800 + Predictions"
num_bins = 1000 #NOTE: I set this to 1,000 because smaller values misrepresent the fit
plot_graph(data_under_800, num_bins, plot_2_weights, title, x_label, y_label)
#Part b - Gamma
plt.plot(dist_pts, gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b),\
        linewidth=2, label='Gamma', color="blue")
#Part c - Generalized Gamma
plt.plot(dist_pts, gen_gamma_fun_pdf(dist_pts, alpha_MLE_c, beta_MLE_c, m_MLE_c),\
        linewidth=2, label="Generalized Gamma", color="green")
#Part d - Generalized Beta 2
plt.plot(dist_pts, gen_beta2_fun_pdf(dist_pts, a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d),\
        linewidth=2, label="Generalized Beta 2", color="red")
plt.legend(loc="upper left")
plt.ylim([0, 0.01])
plt.xlim([0, 800])
plt.show()
#%%
#Problem 2
data = np.loadtxt("/Volumes/GoogleDrive/My Drive/3rd Year/Quarter 2" +\
                  "/Structural Estimation/StructEst_W19/ProblemSets/" +\
                  "PS2/MacroSeries.txt", delimiter=',', usecols=range(4),\
                  dtype=np.float128)
kapital = data[:,1]
wages = data[:,2]
interest = data[:,3]
#%%
#Problem 2 Part a
#Problem 2 Part a - Define Functions
def gen_norm_pdf(xvals, mu, sigma):
        pdf_vals = e ** ( - ( (xvals - mu) ** 2) / (2 * sigma ** 2) )/\
                   (sigma * math.sqrt(2 * math.pi) )
        return pdf_vals

def log_sum_norm(z_vals, mu, sigma):
        log_vals = -( (z_vals - mu) ** 2) / (2 * sigma ** 2) -\
                   np.log(sigma * math.sqrt(2 * math.pi) )
        return log_vals.sum()

def crit_norm_wages(params, *args):
        alpha, rho, mu, sigma = params
        kapital, wages = args
        z_vals = np.log(wages) - np.log(1 - alpha) - alpha * np.log(kapital)
        z_vals_shift = np.roll(z_vals, 1)
        z_vals_shift[0] = mu
        mu_est = rho * z_vals_shift + (1 - rho) * mu
        log_lik_val = log_sum_norm(z_vals, mu_est, sigma)
        return log_lik_val*(-1)
#%%
#Problem 2 Part a - Define Values
alpha_0 = 0.6
rho_0 = 0.6
z_vals = np.log(wages) - np.log(1 - alpha_0) - alpha_0 * np.log(kapital)
mu_0 = z_vals[0]
sigma_0 = 0.15
params_init = np.array([alpha_0, rho_0, mu_0, sigma_0])
#%%
#Problem 2 Part a - Call Functions
results_cstr = opt.minimize(crit_norm_wages, params_init,\
                            args=((kapital, wages)),\
                            method="L-BFGS-B",\
                            bounds=((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10),\
                            (1e-10, None), (1e-10, None)))
alpha_MLE_a, rho_MLE_a, mu_MLE_a, sigma_MLE_a = results_cstr.x
log_lik_norm = -results_cstr.fun
#%%
#Problem 2 Part a - Output Results
print("alpha_MLE=", alpha_MLE_a, " rho_MLE=", rho_MLE_a,\
      " mu_MLE=", mu_MLE_a, " sigma_MLE=", sigma_MLE_a)
print("Maximize Log Likelihood: ", log_lik_norm)
#%%
#Problem 2 Part a - Define Hessian Values
vcv_mle = (results_cstr.hess_inv).matmat(np.eye(4))
stderr_alpha_mle = math.sqrt(vcv_mle[0,0])
stderr_rho_mle = math.sqrt(vcv_mle[1,1])
stderr_mu_mle = math.sqrt(vcv_mle[2,2])
stderr_sig_mle = math.sqrt(vcv_mle[3,3])
#%%
#Problem 2 Part a - Output Hessian Results
print("VCV(MLE) = ", vcv_mle)
print("Standard error for alpha estimate = ", stderr_alpha_mle)
print("Standard error for rho estimate = ", stderr_rho_mle)
print("Standard error for mu estimate = ", stderr_mu_mle)
print("Standard error for sigma estimate = ", stderr_sig_mle)
#%%
#Problem 2 Part b
#Problem 2 Part b - Define Functions
def crit_norm_interest(params, *args):
        alpha, rho, mu, sigma = params
        kapital, interest = args
        z_vals = np.log(interest) - np.log(alpha) - (1 - alpha) * np.log(kapital)
        z_vals_shift = np.roll(z_vals, 1)
        z_vals_shift[0] = mu
        mu_est = rho * z_vals_shift + (1 - rho) * mu
        log_lik_val = log_sum_norm(z_vals, mu_est, sigma)
        return -log_lik_val
#%%
#Problem 2 Part b - Define Values
alpha_0 = 0.6
rho_0 = 0.6
z_vals = np.log(interest) - np.log(alpha_0) - (1 - alpha_0) * np.log(kapital)
mu_0 = z_vals[0]
sigma_0 = 0.15
params_init = np.array([alpha_0, rho_0, mu_0, sigma_0])
#%%
#Problem 2 Part b - Call Functions
results_cstr = opt.minimize(crit_norm_interest, params_init,\
                            args=((kapital, wages)),\
                            method="L-BFGS-B",\
                            bounds=((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10),\
                            (1e-10, None), (1e-10, None)))
alpha_MLE_b, rho_MLE_b, mu_MLE_b, sigma_MLE_b = results_cstr.x
log_lik_norm = -results_cstr.fun
print("alpha_MLE=", alpha_MLE_b, " rho_MLE=", rho_MLE_b,\
      " mu_MLE=", mu_MLE_b, " sigma_MLE=", sigma_MLE_b)
print("Maximize Log Likelihood: ", log_lik_norm)
#%%
#Problem 2 Part b - Define Hessian Values
vcv_mle = (results_cstr.hess_inv).matmat(np.eye(4))
stderr_alpha_mle = math.sqrt(vcv_mle[0,0])
stderr_rho_mle = math.sqrt(vcv_mle[1,1])
stderr_mu_mle = math.sqrt(vcv_mle[2,2])
stderr_sig_mle = math.sqrt(vcv_mle[3,3])
#%%
#Problem 2 Part b - Output Hessian Results
print("VCV(MLE) = ", vcv_mle)
print("Standard error for alpha estimate = ", stderr_alpha_mle)
print("Standard error for rho estimate = ", stderr_rho_mle)
print("Standard error for mu estimate = ", stderr_mu_mle)
print("Standard error for sigma estimate = ", stderr_sig_mle)
#%%
#Problem 2 Part c
#Problem 2 Part c - Define Functions
def integrand_2_c(x, mu, sigma):
        return gen_norm_pdf(x, mu, sigma)
#%%
#Problem 2 Part c - Define Values
k_t = 7500000
z_t_prev = 10
z_star = -np.log(alpha_MLE_a) - (alpha_MLE_a - 1) * np.log(k_t)
mu = rho_MLE_a * z_t_prev + (1 - rho_MLE_a) * mu_MLE_a
#%%
#Problem 2 Part c - Call Functions
pr_rt_gt_1 = integrate.quad(integrand_2_c, z_star, math.inf,\
                            args=(mu, sigma_MLE_a))[0]
#%%
#Problem 2 Part c - Output Results
print("Probability: ", pr_rt_gt_1)
