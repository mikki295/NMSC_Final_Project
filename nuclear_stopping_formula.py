#####################################################################################
#                                                                                   #
#   Library to calculate the universal nuclear stopping power                       #
#   for project 1 in NMSC                                                           #
#                                                                                   #
#   Calculate the stopping power with universal nuclear stopping power formula      #
#   using equations 14,15 and 16 from the project 1 description.                    #
#   More info of the formula from                                                   #
#                                                                                   #
#   http://en.wikipedia.org/wiki/Stopping_power_(particle_radiation)                #
#                                                                                   #
#   Mikael De Meulder 18.5.2021                                                     #
#                                                                                   #
#####################################################################################

# Convetions used
#
# Z1,M1 = Atomic number and mass of the atom being hit
# Z2,M2 = Atomic number and mass of the projectile atom being shot
# Elab  = energy on laboratory coordinates 
#
# Energy units in keV 

import numpy as np 

def epsilon(Z1,M1,Z2,M2,Elab):
    '''See equation (16) in the project description'''
    num = 32.53*M2*Elab
    dem = Z1*Z2*(M1+M2)*(Z1**(0.23) + Z2**(0.23))
    return  num / dem

def s_n(eps):
    '''See equation (15) in the project description'''
    if (eps <= 30):
        num = np.log(1.+1.138*eps)
        dem = 2*(eps + 0.01321*eps**0.21226 + 0.19593*eps**0.5)
        return num / dem
    else:
        return np.log(eps)/(2*eps)


def Sn(Z1,M1,Z2,M2,Elab):
    '''See equation (14) in the project
    description'''
    eps = epsilon(Z1,M1,Z2,M2,Elab)
    sn = s_n(eps)
    
    num = (8.462e-15) *Z1*Z2*M1
    dem = (M1+M2)*(Z1**(0.23) + Z2**(0.23))

    return (num/dem)*sn