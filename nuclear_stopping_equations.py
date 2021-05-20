#################################################################################
#                                                                               #  
#   Library to calculate numerically nuclear stopping power                     #
#   for project 1 in NMSC                                                       #
#                                                                               #
#   This library uses all necessary equations from project 1 description        #
#   used to calculate the nuclear stopping power. Function g_squared is         #
#   the g(r) function but taken the square from in order to solve the           #
#   root. All length units in Å and energy units in keV                         #
#                                                                               #
#   Mikael De Meulder 18.5.2021                                                 #
#                                                                               #
#################################################################################

#   Convetions used
#   
#   Z1,M1 = Atomic number and mass of the target atom being hit
#   Z2,M2 = Atomic number and mass of the projectile atom being shot
#   Elab  = energy in laboratory coordinates  
#   Ecom  = energy in center of mass coordinates
#   r     = distance of projectile and atom
#   r_min = distance of minimum approach of ion and atom
#   b     = collision parameter
#   b_max = maximum value of collision parameter
#
#   Length units in Å
#   Energy units in keV

import numpy as np
import math_library

def g_squared(r,b,Ecom,Z1,Z2):
    '''g(r)^2 See equation (3) in project description'''
    Vr = screened_Coulomb(Z1,Z2,r)
    return (1-(b/r)**2 - Vr/Ecom)


def screened_Coulomb(Z1,Z2,r): 
    '''See equation (5) from project description'''
    au = screening_length(Z1,Z2)
    Phi = screening_function(r/au) 
    coeff = 0.01439962 # keV * Å

    return coeff * ((Z1*Z2)/(r)) * Phi

def screening_length(Z1,Z2):
    ''' See equation (6) in project description.'''
    return ((0.46843)/(Z1**0.23 + Z2**0.23))

def screening_function(x):
    '''See equation (7) in project description. Parameter 
    x is the projectile penetration distance'''

    a = np.array([0.1818,0.5099,0.2802,0.02817])
    b = np.array([3.2,0.9423,0.4028,0.2016])

    return np.sum(a*np.exp(-b*x))

def Sn(Z1,M1,Z2,M2,b_max,Elab,n):
    '''See equation (8) in project description.'''
    a = 0
    b = b_max
    y = gamma(M1,M2)
    coeff = 2*np.pi*y*Elab
    Ecom = Elab_to_Ecom(M1,M2,Elab)

    ans =  math_library.integrate(a,b,n,sin_square, Z1,Z2,Ecom,n)
    return coeff * ans

def sin_square(b,Z1,Z2,Ecom,n):
    '''Integral part of equation (8) in project description'''
    O = theta(Z1,Z2,b,Ecom,n)
    return ((np.sin(0.5*O))**2) * b

def gamma(M1,M2):
    '''See equation (9) in project description'''
    return ((4*M1*M2)/((M1+M2)**2))

def Elab_to_Ecom(M1,M2,Elab):
    '''See equation (10) in project description'''
    return (M2/(M1+M2))*Elab

def Ecom_to_Elab(M1,M2,Ecom):
    '''See equation (10) in project description'''
    return ((M1+M2)/M2)*Ecom

def theta(Z1,Z2,b_max,Ecom,n):
    '''See equation (12) in project description'''
    a = 0
    b = 1
    rmin = math_library.solve_root(g_squared,1e-8,10,b_max,Ecom,Z1,Z2)
    O = np.pi - 4*b_max * math_library.integrate(a,b,n,F,b_max,rmin,Ecom,Z1,Z2)
    return O

def F(u,b,rmin,Ecom,Z1,Z2):
    '''F(u) of scattering scattering integral where the variables has been changed.
    See equation (13) in project description'''


    Vrmin = screened_Coulomb(Z1,Z2,rmin)
    Vumin = screened_Coulomb(Z1,Z2,rmin/(1-u**2))
    left = b**2 * (2-u**2)
    right = ((rmin**2) / (u**2 * Ecom))

    return (left + right*(Vrmin - Vumin))**(-1/2.)