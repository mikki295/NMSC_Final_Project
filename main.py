from builtins import len, print
import numpy as np
import matplotlib.pyplot as plt
import nuclear_stopping_formula as nsf
import nuclear_stopping_equations as nse
import time
from sklearn.metrics import mean_squared_error as mse


def const1(n):
    '''Returns the constants of part 1 a) H, Si. Atomic number and mass returned
    as tuples. n is the number of energies calculated.'''
    return (1, 1.007825,'H'), (14, 27.976927,'Si'), np.logspace(np.log10(0.01),np.log10(5000),num=n)

def const2(n):
    '''Returns the constants of part 1 b) Si, Au. Atomic number and mass returned
    as tuples. n is the number of energies calculated.'''
    
    return (14, 27.976927,'Si'), (79, 196.966543,'Au'), np.logspace(np.log10(0.01),np.log10(5000),num=n)

def calculate_stopping_powers(case,b_max,n_weigths):
    n = 200
    r = np.linspace(0.01,2*b_max,n)
    
    SP = np.zeros(n)
    USP = np.zeros(n)
    theta = np.zeros(n)

    projectile, atom, Elab = const2(n)
    Z1,M1,X1 = atom
    Z2,M2,X2 = projectile

    t = time.time()
    for i in range(n):
        SP[i] = nse.Sn(Z1,M1,Z2,M2,b_max,Elab[i],n_weigths)* 1e-13
        USP[i] = nsf.Sn(Z1,M1,Z2,M2,Elab[i])
        #USP[i] = nsf.fit(Elab[i],Z1,Z2,M1,M2)
    computing_time = time.time()-t

    print('Computation time: {:.2f} s'.format(computing_time))


    # ------------- Plotting ------------- #
    #plt.plot(Elab,SP,label='Numerical Stopping Power')
    #plt.plot(Elab,USP,'--',label='Universal Formula')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.xlabel(r'$E_{lab} / [keV]$')
    #plt.ylabel(r'$S_n (E_{lab}) / [eV cm^2]$')
    #plt.legend(loc='best')
    #title = r'$^{}{} \rightarrow ^{{}}{}$'.format({Z2},X2,{Z1},X1)
    #plt.title(title)
    #plt.show()

    # Mean squared "error" between numerically calculated stopping power
    # and by universal stopping power formula
    difference = mse(SP,USP)

    return difference,computing_time

def potential_curves():
    r = np.linspace(0.0001,10,1000)
    pot = np.zeros((2,len(r)))
    case = [[1,14],[14,79]]

    for i in range(len(r)):
        for j in (0,1):
            pot[j][i] = nse.screened_Coulomb(case[j][1],case[j][0],r[i])


    plt.plot(r,pot[0],label=r'$^1 \mathrm{H} \rightarrow ^{28} \mathrm{Si}$')
    plt.plot(r,pot[1],label=r'$^{28} \mathrm{Si} \rightarrow ^{197} \mathrm{Au}$')
    plt.yscale('log')
    plt.ylabel('V(r) / [keV]')
    plt.xlabel('r / [$\AA$]')
    plt.title('Screened Coulomb Potential')
    plt.grid('True')
    plt.legend(loc='best')
    plt.show()

def computation_accuracy_and_time():
    n = 15
    b_max = 4 # Å

    weights = np.arange(10,101,n)
    print(weights)
    dif1 = np.zeros(len(weights))
    dif2 = np.zeros(len(weights))

    comp_t1 = np.zeros(len(weights))
    comp_t2 = np.zeros(len(weights))

    for i in range(len(weights)):
        dif1[i],comp_t1[i] = calculate_stopping_powers(const1,b_max,weights[i])
        dif2[i],comp_t2[i] = calculate_stopping_powers(const2,b_max,weights[i])


    fig, ax1 = plt.subplots()

    ax1.plot(weights,dif1,c='#1f77b4',marker='o',linestyle='None')
    ax1.plot(weights,dif2,c='#1f77b4',marker='x',linestyle='None')
    ax1.set_xlabel('Weights')
    ax1.set_ylabel('MSE')
    ax1.tick_params('y',colors='#1f77b4')

    ax2 = ax1.twinx()
    ax2.plot(weights,comp_t1,c='#ff7f0e',marker='o',linestyle='None')
    ax2.plot(weights,comp_t2,c='#ff7f0e',marker='x',linestyle='None')
    ax2.set_ylabel('Computation time / [s]')
    ax2.tick_params('y',colors='#ff7f0e')
    fig.tight_layout()
    plt.grid('True')
    plt.show()

def main():
    #potential_curves()

    n = 8
    b_max = 4 # Å

    computation_accuracy_and_time()


if __name__ == "__main__":
    main()
