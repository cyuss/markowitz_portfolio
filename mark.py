#!/usr/bin/python

import pandas as pd
import numpy as np
import cvxopt as opt
import cvxopt.solvers as optsolvers
import matplotlib.pyplot as plt
from math import sqrt
#from ggplot import *
#from scipy.stats import gmean
import warnings


# display a section
def section(caption):
    print('\n\n' + str(caption))
    print('-' * len(caption)) 

def load_data(fileName):
    df = pd.read_csv(fileName, sep = ",")
    return df

def rates_return(prices) :
    #prices = prices.as_matrix()
    dim = prices.shape
    nb_rows, nb_cols = dim[0], dim[1] 
    rate_ret = np.zeros((nb_rows - 1, nb_cols))
    # p[i, j] = (p[i+1, j] - p[i, j]) / p[i, j]
    for i in range(0, nb_rows-1) :
        for j in range(0, nb_cols) :
            rate_ret[i, j] = (prices[i+1, j] - prices[i, j]) / prices[i, j]
    return rate_ret

def markwoitz_portfolio(prices, target_ret = 0.0006, allow_short = True) :
    prices = prices.as_matrix()

    # rates of returns
    rate_rets = rates_return(prices)
    
    # covariance matrix
    cov_mat = np.cov(rate_rets.T)
    
    # expected returns with arithmetic mean
    exp_rets = pd.Series(np.mean(rate_rets, axis = 0))
    
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
    #return weights, ret, risk, cov_mat, exp_rets
    return cov_mat, exp_rets

df = load_data("prices.csv")
d = df.head(2518)
d = d[['AA', 'AXP', 'CAT', 'DD']]

#k, x, y, sd, me = markwoitz_portfolio(d, allow_short = False)
sd, me = markwoitz_portfolio(d, allow_short = False)
"""section("weights")
print k
print k.sum()
lx = []
ly = []

space = np.arange(.0001, .0008, .000001)
#print len(space)
N = 10
for r in space :
    k, x, y = markwoitz_portfolio(d, target_ret = r, allow_short = True)
    lx.append(x*10000)
    ly.append(y*100)

print len(ly)
z = zip(lx, ly)
#for j in range(0, 10) :#
	#print z[j]
#donnees = {'x': pd.Series(lx), 'y': pd.Series(ly)}
#donnees = pd.DataFrame(donnees)
#print donnees.head()
#pl = ggplot(aes(x = 'x', y = 'y'), data = donnees) + geom_point() + ggtitle('risk ~ return') + xlab("risk") + ylab("return")

#print pl
plt.scatter(lx, ly, marker = 'o')
plt.xlabel('std')
plt.ylabel('mean')
plt.show()"""

n_assets = 4
n_obs = 1000
return_vec = np.random.randn(n_assets, n_obs)

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns, me, sd):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    #p = np.asmatrix(np.mean(returns, axis=1))
    p = me
    w = np.asmatrix(rand_weights(returns.shape[0]))
    #C = np.asmatrix(np.cov(returns))
    C = sd
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_vec, me, sd) 
    for _ in xrange(n_portfolios)
])

print "----- means -------"
print means[range(1, 5)]
print "----- stds -------"
print stds[range(1, 5)]

plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()