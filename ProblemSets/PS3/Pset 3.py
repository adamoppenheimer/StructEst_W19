#%%
# MACS 402 - Structural Analysis - Dr. Evans - Problem Set 3
# DATE: February 6th, 2019
# AUTHOR: Adam Oppenheimer
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
import scipy.integrate as integrate
import math
from math import e
DATA_DIR = os.path.abspath("")
#%%
#Problem 1
INCOMES_FILENAME = os.path.join(DATA_DIR, "data/usincmoms.txt")
incomes = np.loadtxt(INCOMES_FILENAME)
incomes[:,1] = incomes[:,1] / 1000
#%%
#Problem 1 Part a
incomes_graph = np.copy(incomes)
incomes_graph[40,0] = incomes_graph[40,0]/10
incomes_graph[41,0] = incomes_graph[41,0]/20
width = 5 * np.ones(42)
width[40] = 50
width[41] = 100
plt.bar(x=incomes_graph[:,1], height=incomes_graph[:,0], width=width)
plt.title("Percent of Population by Income")
plt.xlabel("Income ($1000s)")
plt.ylabel("Percent of Population")
#plt.show()
#%%
#Problem 1 Part b
def ln_fun_pdf(xvals, mu, sigma):
    pdf_vals = e ** ( - (np.log(xvals) - mu) ** 2 / (2 * sigma ** 2) ) /\
            (xvals * sigma * (2 * math.pi) ** 0.5)
    return pdf_vals

def moments_model(mu, sigma):
    divisions = np.ones(43)
    divisions_beginning = np.linspace(0,200,41)
    divisions[0:41] = divisions_beginning
    divisions[0] = 1e-10
    divisions[41] = 250
    divisions[42] = 350
    integrals = []

    prob_notcut = integrate.quad(ln_fun_pdf, 1e-10,\
                                        350,\
                                        args=(mu, sigma))[0]
    for i in range(len(divisions) - 1):
        integrals.append(integrate.quad(ln_fun_pdf,\
                                        divisions[i],\
                                        divisions[i + 1],\
                                        args=(mu, sigma))[0]\
                                        / prob_notcut)
    return np.array(integrals)

#%%
def err_vec(xvals, mu, sigma, simple):
    moms_data = xvals[:,1]
    moms_model = moments_model(mu, sigma)
    #print("Mu: ", mu)
    #print("Sigma: ", sigma)
    if simple:
        err_vector = moms_data - moms_model
    else:
        #import pandas as pd
        #temp = pd.DataFrame(moms_model)
        #temp = temp[temp != 0]
        #if temp.isnull().values.any():
        #    print("VALUE IS 0!!!!")
        #    print("Mu is: ", mu)
        #    print("Sigma is: ", sigma)
        err_vector = (moms_model - moms_data) / moms_data
    return err_vector

def crit(params, *args):
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = err.T @ W @ err
    return crit_val
#%%
income = incomes[:,0] * incomes[:,1]
avg_inc = income.sum()
mu_init = np.log(avg_inc)
#sig_init = 7
#fun: 0.99997185359359
#SUCCESS
#sig_init = 5
#fun: 0.99997104024862
#FAILURE
sig_init = 4
#fun: 0.9999718535439717
#SUCCESS
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(42)
W_hat = W_hat * incomes[:,0]
gmm_args = (incomes, W_hat)
results = opt.minimize(crit, params_init, args=(gmm_args), tol=1e-20,
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
print(results)
#%%
centers = incomes[:,1]

vals = moments_model(mu_GMM1, sig_GMM1)#ln_fun_pdf(centers, mu_GMM1, sig_GMM1)
vals[40] = vals[40]/10
vals[41] = vals[41]/20
print(vals)

plt.bar(x=incomes_graph[:,1], height=incomes_graph[:,0], width=width)

plt.scatter(centers, vals,\
        linewidth=2, label="Log Normal Function", color="r")

plt.title("Percent of Population by Income")
plt.xlabel("Income ($1000s)")
plt.ylabel("Percent of Population")
plt.show()
#%%
#Problem 1 Part c
def gamma_fun_pdf(xvals, alpha, beta):
        pdf_vals = ( (xvals / beta) ** alpha * e ** ( - xvals / beta ) )/\
                   ( xvals * math.gamma(alpha) )
        return pdf_vals

def moments_model_2(alpha, beta):
    divisions = np.ones(43)
    divisions_beginning = np.linspace(0,200,41)
    divisions[0:41] = divisions_beginning
    divisions[0] = 1e-10
    divisions[41] = 250
    divisions[42] = 350
    integrals = []

    prob_notcut = integrate.quad(gamma_fun_pdf, 1e-10,\
                                        350,\
                                        args=(alpha, beta))[0]
    if prob_notcut == 0:
        return [0]*42
    for i in range(len(divisions) - 1):
        value = integrate.quad(gamma_fun_pdf,\
                                        divisions[i],\
                                        divisions[i + 1],\
                                        args=(alpha, beta))[0]
        integrals.append(value / prob_notcut)
    return np.array(integrals)

#%%
def err_vec_2(xvals, alpha, beta, simple):
    moms_data = xvals[:,1]
    moms_model = moments_model_2(alpha, beta)
    if simple:
        err_vector = moms_data - moms_model
    else:
        err_vector = (moms_model - moms_data) / moms_data
    return err_vector

def crit_2(params, *args):
    alpha, beta = params
    xvals, W = args
    err = err_vec_2(xvals, alpha, beta, simple=False)
    crit_val = err.T @ W @ err
    return crit_val

income = incomes[:,0] * incomes[:,1]
avg_inc = income.sum()
alpha_init = 3
beta_init = 20000
params_init = np.array([alpha_init, beta_init])
W_hat = np.eye(42)
W_hat = W_hat * incomes[:,0]
gmm_args = (incomes, W_hat)
results = opt.minimize(crit_2, params_init, args=(gmm_args), tol=1e-20,
                        method='L-BFGS-B',\
                        bounds=((1e-10, None), (1e-10, None)))
alpha_GMM1, beta_GMM1 = results.x
print('alpha_GMM1=', alpha_GMM1, ' beta_GMM1=', beta_GMM1)
print(results)
#%%
centers = incomes[:,1]

vals = moments_model_2(alpha_GMM1, beta_GMM1)
vals[40] = vals[40]/10
vals[41] = vals[41]/20
print(vals)

plt.bar(x=incomes_graph[:,1], height=incomes_graph[:,0], width=width)

plt.scatter(centers[1:], vals[1:],\
        linewidth=2, label="Log Normal Function", color="r")

plt.title("Percent of Population by Income")
plt.xlabel("Income ($1000s)")
plt.ylabel("Percent of Population")
plt.show()