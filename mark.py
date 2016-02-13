#!/usr/bin/python

import pandas as pd
import numpy as np
import cvxopt as opt
import cvxopt.solvers as optsolvers
from cvxopt import blas
import matplotlib.pyplot as plt
from math import sqrt
import warnings


# display a section
def section(caption):
    print('\n\n' + str(caption))
    print('-' * len(caption)) 

def load_data(fileName):
    df = pd.read_csv(fileName, sep = ",")
    return df

def rates_return(prices) :
    prices = prices.as_matrix()
    dim = prices.shape
    nb_rows, nb_cols = dim[0], dim[1] 
    rate_ret = np.zeros((nb_rows - 1, nb_cols))
    # p[i, j] = (p[i+1, j] - p[i, j]) / p[i, j]
    for i in range(0, nb_rows-1) :
        for j in range(0, nb_cols) :
            rate_ret[i, j] = (prices[i+1, j] - prices[i, j]) / prices[i, j]
    return rate_ret

def cov_matrix(rates) :
    cov_mat = np.cov(rates.T)
    return cov_mat

def exp_returns(rates) :
    exp_rets = pd.Series(np.mean(rates, axis = 0))
    return exp_rets

def markwoitz_portfolio(rate_rets, cov_mat, exp_rets, target_ret = 0.0006, allow_short = True, lmin = 0, lmax = 1) :
    # matrices conversion : P = covariance
    n = len(cov_mat)
    P = opt.matrix(cov_mat)
    q = opt.matrix(0.0, (n, 1))

    # constraints Gx <= h
    if not allow_short:
        # exp_rets * x >= 1 and x >= 0
        G = opt.matrix(np.vstack((-exp_rets.values, -np.identity(n))))
        h = opt.matrix(np.vstack((-target_ret, +np.zeros((n, 1)))))
    else :
        # exp_rets * x >= 1
        G = opt.matrix(-exp_rets.values).T
        h = opt.matrix(-target_ret)
    
    A = opt.matrix(1.0, (1, n))
    x = opt.matrix(1.0, (n, 1))
    b = opt.matrix(1.0)

    optsolvers.options['show_progress'] = False

    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal' :
        warnings.warn("Convergence problem")
    
    weights = pd.Series(sol['x'])
    ret = (opt.matrix(weights).T * opt.matrix(exp_rets))[0, 0]
    risk = np.sqrt(np.asmatrix(weights) * cov_mat * np.asmatrix(weights).T)
    #risk = [np.sqrt(blas.dot(np.asmatrix(x).T, cov_mat*np.asmatrix(x))) for x in weights]
    #print "here type"
    #print type(risk)
    return weights, ret, risk, cov_mat, exp_rets
    #return cov_mat, exp_rets

def main() :
    # loading data
    df = load_data("prices.csv")
    d = df.head(2518)
    d = d[['AA', 'AXP', 'CAT', 'DD']]

    # compute covariance matrix, rates of returns and expected returns
    rateret = rates_return(d)
    covmat = cov_matrix(rateret)
    exprets = exp_returns(rateret)

    k, x, y, sd, me = markwoitz_portfolio(rateret, covmat, exprets, allow_short = False)

    y = np.asarray(y)
    lx = []
    ly = []

    space = np.arange(.0001, .0008, .000001)

    for r in space :
        k, x, y, t1, t2 = markwoitz_portfolio(rateret, covmat, exprets, target_ret = r, allow_short = True)
        lx.append(x)
        ly.append(y)

    #print len(ly)
    z = zip(lx, ly)

    plt.scatter(ly, lx, marker = 'o')
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.show()

main()