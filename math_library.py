#########################################################
#                                                       #
#   Math library for project 1 in NMSC                  #
#                                                       #
#   Quite short, but easy to extend if necessary.       #
#   Contains integration with Gauss-Legendre            #
#   polynomials and root solving with Brent's method    #
#                                                       #
#   Mikael De Meulder 18.5.2021                         #
#                                                       #
#########################################################

import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import brentq

x100,w100 = np.loadtxt('weights100.txt',unpack=True)

def calc_weigths(n):
    '''Caclulate n Gauss-Legendre polynomial weights. Return the
    precalculated weights if n == 100'''
    if (n == 100):
        return x100,w100
    return roots_legendre(n)

def integrate(a,b,n,f,*args):
    '''Integrate using Gauss-Legendre polynomials. a and b are the lower 
    and upper limits, n is the number of weights and f is the funciton.
    args is all the necessary arguments the funciton needs.'''

    x,w = calc_weigths(n)
    left = (b-a)/2
    right = (a+b)/2

    ans = np.zeros(n)
    for i in range(n):
        ans[i] = w[i]*f(left*x[i] + right,*args)

    return left * np.sum(ans)

def solve_root(f,xmin,xmax,*args):
    '''Solve root using Brent's method. f = function, xmin = lower bound,
    xmax = upper bound, args = arguments the function needs.'''
    return brentq(f,xmin,xmax,args)

def infinity_norm(v1,v2):
    '''Calculate infinity norm'''
    return np.max(np.abs(v1-v2))