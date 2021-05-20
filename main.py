#############################################################################
#                                                                           #
#   Main of Final Project problem 1 in NMSC                                 #
#                                                                           #
#   Project 1 is about calulating numerically the stopping powers           #
#   of H --> Si (case1) and Si --> Au (case2). See README to see            #
#   how the code is structured.                                             #
#                                                                           #
#   Mikael De Meuder 18.5.2021                                              #
#                                                                           #
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

import nuclear_stopping_formula as nsf
import nuclear_stopping_equations as nse
import math_library

def case_values(n):
    '''Returns the constants of part 1 a) H, Si. Atomic number and mass returned
    as tuples. n = 1 or 2 depending which case to calculate.'''
    if (n == 1):
        return (1, 1.007825,'^1H'), (14, 27.976927,'^{28}Si')
    else:
       return (14, 27.976927,'^{28}Si'), (79, 196.966543,'^{197}Au')
    
def calculate_stopping_powers(case,b_max,Elab,n_weigths,plot):
    '''Calculate the stopping power numerically. case = 1 or 2, 1 = H --> Si
    2 = Si --> Au. See function case_values().'''
    n = 200
    
    SP = np.zeros(n)
    USP = np.zeros(n)

    projectile, atom = case_values(case)
    Z1,M1,X1 = atom
    Z2,M2,X2 = projectile

    # Calculate the Stopping power and the universal stopping power
    # and measure the computation time

    t = time.time()
    for i in range(n):
        SP[i] = nse.Sn(Z1,M1,Z2,M2,b_max,Elab[i],n_weigths)* 1e-13 #convert keV * Å^2 to ev * cm^2
        USP[i] = nsf.Sn(Z1,M1,Z2,M2,Elab[i])
    computing_time = time.time()-t
    print('Computation time: {:.2f} s'.format(computing_time))


    # ------------- Plot if true ------------- #
    fontsize = 13
    if (plot):
        plt.plot(Elab,SP,label='Numerical Stopping Power')
        plt.plot(Elab,USP,'--',label='Universal Formula')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$E_{lab}$ / [keV]',fontsize=fontsize)
        plt.ylabel(r'$S_n (E_{lab})$ / [eV cm$^2$]',fontsize=fontsize)
        plt.legend(loc='best')
        title = r'${} \rightarrow {}$'.format(X2,X1)
        plt.title(title,fontsize=fontsize)
        plt.show()

    # Infinity norm between numerically calculated stopping power
    # and by universal stopping power formula
    difference = math_library.infinity_norm(SP,USP)

    return SP,USP,difference,computing_time

def potential_curves(r):
    '''Calculate screened Coulomb potential curve from 0 to r
    between H - Si and Si - Au.'''

    r = np.linspace(0.0001,r,1000)
    pot = np.zeros((2,len(r)))
    case = [[1,14],[14,79]]

    for i in range(len(r)):
        for j in (0,1):
            pot[j][i] = nse.screened_Coulomb(case[j][1],case[j][0],r[i])

    # ------------- Plot ------------- #
    fontsize = 13
    plt.plot(r,pot[0],label=r'$^1 \mathrm{H} \rightarrow ^{28} \mathrm{Si}$')
    plt.plot(r,pot[1],label=r'$^{28} \mathrm{Si} \rightarrow ^{197} \mathrm{Au}$')
    plt.yscale('log')
    plt.ylabel('V(r) / [keV]',fontsize=fontsize)
    plt.xlabel('r / [$\AA$]',fontsize=fontsize)
    plt.title('Screened Coulomb Potential',fontsize=fontsize)
    plt.grid('True')
    plt.legend(loc='best')
    plt.show()

def computation_accuracy_and_time(Elab,b_max):
    '''Comparing the computation accuracy and time.'''
    n = 15  
    weights = np.arange(10,101,n) # Weight nodes
    print("Number of weights: {}".format(weights))
    dif1 = np.zeros(len(weights))
    dif2 = np.zeros(len(weights))

    comp_t1 = np.zeros(len(weights))
    comp_t2 = np.zeros(len(weights))


    # Computing the stopping powers. Case 2 commented out 
    # because it didn't differ significantly from case 1

    for i in range(len(weights)):
        _,_,dif1[i],comp_t1[i] = calculate_stopping_powers(case_values(1),b_max,Elab,weights[i],False)
        #_,_,dif2[i],comp_t2[i] = calculate_stopping_powers(case_values(2),b_max,Elab,weights[i],False)

    # ------------- Plot ------------- #
    fig, ax1 = plt.subplots()

    ax1.plot(weights,dif1,c='#1f77b4',marker='o',linestyle='None')
    #ax1.plot(weights,dif2,c='#1f77b4',marker='x',linestyle='None')
    ax1.set_xlabel('Number of weights')
    ax1.set_ylabel(r'$L_{\infty}$')
    ax1.tick_params('y',colors='#1f77b4')

    ax2 = ax1.twinx()
    ax2.plot(weights,comp_t1,c='#ff7f0e',marker='o',linestyle='None')
    #ax2.plot(weights,comp_t2,c='#ff7f0e',marker='x',linestyle='None')
    ax2.set_ylabel('Computation time / [s]')
    ax2.tick_params('y',colors='#ff7f0e')
    fig.tight_layout()
    plt.grid('True')
    plt.show()

def main():
    # Calculate the potential curve to deternime reasonable b_max
    r = 10 # Å
    potential_curves(r)

    # Determine the "Sweet spot" of accuracy and computation time
    # for number of nodes in Gauss-Legendre quadrature. I settled with 100
    # because the computation is only necessary to do once
     
    #computation_accuracy_and_time(Elab,6) # 6 = b_max

    # The actual computation
    n = 30
    b_max = 6 # Å
    Elab = np.logspace(np.log10(0.01),np.log10(5000),num=200)
    SP1,USP1,_,_ = calculate_stopping_powers(1,b_max,Elab,n,True)
    SP2,USP2,_,_ = calculate_stopping_powers(2,b_max,Elab,n,True)


if __name__ == "__main__":
    main()

# Around Elab = 1100 keV is when epsilon > 30