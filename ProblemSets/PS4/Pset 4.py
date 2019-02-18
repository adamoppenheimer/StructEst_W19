#%%
# MACS 402 - Structural Analysis - Dr. Evans - Problem Set 4
# DATE: February 16th, 2019
# AUTHOR: Adam Oppenheimer
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as integrate
import math
from math import e
DATA_DIR = os.path.abspath("")
DATA_DIR = "/Volumes/GoogleDrive/My Drive/3rd Year/Quarter 2/Structural Estimation/StructEst_W19/ProblemSets/PS4"
#%%
#Problem 1
DATA_FILENAME = os.path.join(DATA_DIR, "data/NewMacroSeries.txt")
DATA_FILENAME = DATA_DIR + "/data/NewMacroSeries.txt"
data = np.loadtxt(DATA_FILENAME, delimiter=',',\
                    usecols=range(5), dtype=np.float128)
consumption = data[:,0]
kapital = data[:,1]
wages = data[:,2]
interest = data[:,3]
output = data[:,4]
#%%
#Problem 1 Part a - Define Functions
def calc_corr(val1, val2):
    rows = min(val1.shape[0], val2.shape[0])
    cols = val1.shape[1]
    corrs = []
    for i in range(cols):
        corrs.append(np.corrcoef(val1[:rows, i],\
                        val2[:rows, i])[1,0])
    return np.array(corrs)

def calc_moms(consumption, kapital, output):
    if consumption.ndim == 1:
        mean_c = consumption.mean()
        mean_k = kapital.mean()
        mean_c_y = (consumption / output[:consumption.shape[0]]).mean()
        var_y = np.var(output)
        corr_c = np.corrcoef(consumption[1:], consumption[:-1])[1,0]
        corr_c_k = np.corrcoef(consumption, kapital[:consumption.shape[0]])[1,0]
    elif consumption.ndim == 2:
        mean_c = consumption.mean(axis=0)
        mean_k = kapital.mean(axis=0)
        mean_c_y = (consumption / output[:consumption.shape[0]]).mean(axis=0)
        var_y = np.var(output, axis=0)
        corr_c = calc_corr(consumption[1:,:], consumption[:-1,:])
        corr_c_k = calc_corr(consumption, kapital)
    moments = np.array([mean_c, mean_k,\
                        mean_c_y, var_y, corr_c, corr_c_k])
    return moments

def norm_draws(unif_vals, sigma):
    tnorm_draws = sts.norm.ppf(unif_vals, loc=0, scale=sigma)
    return tnorm_draws

def z_draws(e_vals, rho, mu):
    T, sims = e_vals.shape
    z_vals = []
    for s in range(sims):
        z_s = [mu] #We have z_0 = mu
        for t in range(T - 1):
            new_z = rho * z_s[t] + (1 - rho) * mu + e_vals[t, s]
            z_s.append(new_z)
        z_vals.append(z_s)
    z_vals = np.array(z_vals)
    return z_vals.T

def k_draws(z_vals, alpha, beta, k_1):
    T, sims = z_vals.shape
    k_vals = []
    for s in range(sims):
        k_s = [k_1] #We have k_1 = mean(k)
        for t in range(T - 1):
            new_k = alpha * beta * e ** z_vals[t, s] * k_s[t] ** alpha
            k_s.append(new_k)
        k_vals.append(k_s)
    k_vals = np.array(k_vals)
    return k_vals.T

def pred_vals(S, k_1, alpha, beta, rho, mu, sigma):
    e_vals = norm_draws(S, sigma)
    z_vals = z_draws(e_vals, rho, mu)
    k_vals = k_draws(z_vals, alpha, beta, k_1)
    w_vals = (1 - alpha) * e ** z_vals * k_vals ** alpha
    r_vals = alpha * e ** z_vals * k_vals ** (alpha - 1)
    c_vals = w_vals[:-1,:] + r_vals[:-1,:] * k_vals[:-1,:] - k_vals[1:,:]
    y_vals = e ** z_vals * k_vals ** alpha
    return c_vals, k_vals, y_vals

def err_vec_a(moms_data, S, k_1, beta, params, simple=False):
    alpha, rho, mu, sigma = params
    c_pred, k_pred, y_pred = pred_vals(S, k_1, alpha, beta, rho, mu, sigma)
    mean_c, mean_k, mean_c_y, var_y,\
        corr_c, corr_c_k = calc_moms(c_pred, k_pred, y_pred)
    mean_c = mean_c.mean()
    mean_k = mean_k.mean()
    mean_c_y = mean_c_y.mean()
    var_y = var_y.mean()
    corr_c = list(corr_c)
    corr_c = sum(corr_c) / len(corr_c)
    corr_c_k = list(corr_c_k)
    corr_c_k = sum(corr_c_k) / len(corr_c_k)
    moms_model = np.array([mean_c, mean_k, mean_c_y,\
                            var_y, corr_c, corr_c_k ])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    return err_vec

def crit_a(params, *args):
    alpha, rho, mu, sigma = params
    moms_data, k_1, beta, S, W = args
    global iteration
    iteration += 1
    print("Iteration ", iteration)
    err = err_vec_a(moms_data, S, k_1, beta, params, simple=False)
    crit_val = err.T @ W @ err
    print("Crit val: ", crit_val)
    return crit_val
#%%
#Problem 1 Part a - Define Values
alpha_0 = 0.4
rho_0 = 0.9
mu_0 = 9
sigma_0 = 0.04
params_init_a = np.array([alpha_0, rho_0, mu_0, sigma_0])

moms_data = calc_moms(consumption, kapital, output)
k_1 = kapital.mean()
beta = 0.99
S = np.random.rand(100, 1000)
W_hat_a = np.eye(6)
smm_args_a = (moms_data, k_1, beta, S, W_hat_a)
#%%
#Problem 1 Part a - Call Functions
iteration = 0 #Note it takes roughly 250-300 iterations to solve
results_a = opt.minimize(crit_a, params_init_a, args=(smm_args_a),
                          method='L-BFGS-B',
                          bounds=((0.01, 0.9), (-0.99, 0.99),\
                                  (5, 14), (0.01, 1.1)))
#Bounds for alpha changed to (0.01, 0.9) to prevent overflow errors
#This is because alpha > 0.9 caused capital to depreciate too slowly
alpha_SMM_a, rho_SMM_a, mu_SMM_a, sigma_SMM_a = results_a.x
#%%
#Problem 1 Part a - Output Results
print("Alpha_SMM_a=", alpha_SMM_a, " Rho_SMM_a=", rho_SMM_a,\
    "Mu_SMM_a=", mu_SMM_a, "Sigma_SMM_a=", sigma_SMM_a)
c_pred, k_pred, y_pred = pred_vals(S, k_1, alpha_SMM_a, beta,\
                                    rho_SMM_a, mu_SMM_a, sigma_SMM_a)
moms_err = err_vec_a(moms_data, S, k_1, beta, results_a.x, simple=False)
print("Moment Differences (Percent):")
print(moms_err)
print("Criterion Function Value:")
print(results_a.fun)
#%%
#Problem 1 Part a - Define Jacobian Error Function
def Jac_err_a(moms_data, S, k_1, beta, params, simple=False):
    Jac_err = np.zeros((6, 4))
    h_params = 1e-4 * params
    for i in range(4):
        h_mu = h_params[i]
        params_plus = params.copy()
        params_plus[i] += h_mu
        params_minus = params.copy()
        params_minus[i] -= h_mu
        Jac_err[:,i] =\
             ((err_vec_a(moms_data, S, k_1, beta, params_plus, simple=False) -\
             err_vec_a(moms_data, S, k_1, beta, params_minus, simple=False))/\
             (2 * h_mu)).flatten()
    return Jac_err
#%%
#Problem 1 Part a - Call Jacobian Error Function
params_SMM_a = np.array(results_a.x)
d_err_a = Jac_err_a(moms_data, S, k_1, beta, params_SMM_a, simple=False)
#%%
#Problem 1 Part a - Output Std. Errors
print(d_err_a)
print(W_hat_a)
SigHat_a = (1 / S.shape[1]) * lin.inv(d_err_a.T @ W_hat_a @ d_err_a)
print(SigHat_a)
print('Std. err. alpha_hat_a=', np.sqrt(SigHat_a[0, 0]))
print('Std. err. rho_hat_a=', np.sqrt(SigHat_a[1, 1]))
print('Std. err. mu_hat_a=', np.sqrt(SigHat_a[2, 2]))
print('Std. err. sigma_hat_a=', np.sqrt(SigHat_a[3, 3]))
#%%
#Problem 1 Part b
#Problem 1 Part b - Define Error Matrix Function
def get_error_matrix(moms_data, S, k_1, beta, params, simple=False):
    R_shape = len(moms_data)
    S_shape = S.shape[1]
    Err_mat = np.zeros((R_shape, S_shape))

    alpha, rho, mu, sigma = params
    c_pred, k_pred, y_pred = pred_vals(S, k_1, alpha, beta, rho, mu, sigma)
    moms_pred = calc_moms(c_pred, k_pred, y_pred)
    if simple:
        for i in range(len(moms_pred)):
            Err_mat[i, :] = moms_pred[i,:] - moms_data[i]
    else:
        for i in range(len(moms_pred)):
            Err_mat[i, :] = (moms_pred[i,:] - moms_data[i])\
                            / moms_data[i]
    
    return Err_mat
#%%
#Problem 1 Part b - Find New Weighting Matrix
error_matrix = get_error_matrix(moms_data, S, k_1, beta, results_a.x, simple=False)
VCV = (1 / S.shape[1]) * (error_matrix @ error_matrix.T)
print(VCV)
W_hat_b = lin.inv(VCV)
print(W_hat_b)
#%%
#Problem 1 Part b - Define Updated Values
params_init_b = params_SMM_a
smm_args_b = (moms_data, k_1, beta, S, W_hat_b)
#%%
#Problem 1 Part b - Call Functions
iteration = 0
results_b = opt.minimize(crit_a, params_init_b, args=(smm_args_b),
                          method='L-BFGS-B',
                          bounds=((0.01, 0.9), (-0.99, 0.99),\
                                  (5, 14), (0.01, 1.1)))
alpha_SMM_b, rho_SMM_b, mu_SMM_b, sigma_SMM_b = results_b.x
#%%
#Problem 1 Part b - Output Results
print("Alpha_SMM_b=", alpha_SMM_b, " Rho_SMM_b=", rho_SMM_b,\
    "Mu_SMM_b=", mu_SMM_b, "Sigma_SMM_b=", sigma_SMM_b)
c_pred, k_pred, y_pred = pred_vals(S, k_1, alpha_SMM_b, beta,\
                                    rho_SMM_b, mu_SMM_b, sigma_SMM_b)
moms_err = err_vec_a(moms_data, S, k_1, beta, results_b.x, simple=False)
print("Moment Differences (Percent):")
print(moms_err)
print("Criterion Function Value:")
print(results_b.fun)
#%%
#Problem 1 Part b - Call Jacobian Error Function
params_SMM_b = np.array(results_b.x)
d_err_b = Jac_err_a(moms_data, S, k_1, beta, params_SMM_b, simple=False)
#%%
#Problem 1 Part b - Output Std. Errors
print(d_err_b)
print(W_hat_b)
SigHat_b = (1 / S.shape[1]) * lin.pinv(d_err_b.T @ W_hat_b @ d_err_b)
print(SigHat_b)
print('Std. err. alpha_hat_b=', np.sqrt(SigHat_b[0, 0]))
print('Std. err. rho_hat_b=', np.sqrt(SigHat_b[1, 1]))
print('Std. err. mu_hat_b=', np.sqrt(SigHat_b[2, 2]))
print('Std. err. sigma_hat_b=', np.sqrt(SigHat_b[3, 3]))
