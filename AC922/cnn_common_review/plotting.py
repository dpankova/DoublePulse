# Plotting functions for tau ID CNN work 
#
# Aaron Fienberg
# April 2020


import matplotlib.pyplot as plt
import numpy as np
from .util import per_year_past_cut

def plot_history(history, name='training_history'):
    for quantity, vals in history.items():
        if quantity != 'lr':
            plt.plot(vals, label=quantity)
    
    plt.legend(fontsize=16)
    plt.grid()
    
    if not name.endswith('.pdf'):
        name += '.pdf'
    
    plt.savefig(f'{name}', bbox='tight')
    plt.clf()
    

def n_per_year_plot(tables, files_per_flavor, name='n_per_year'):
    cut_vals = np.linspace(0, 1, 101)
    n_past = [[per_year_past_cut(table, cut, n) for cut in cut_vals] 
               for table, n in zip(tables, files_per_flavor)]
    
    orig_size = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = 8, 4
    
    for n, label in zip(n_past, [r'$\nu_{e}$', r'$\nu_{\tau}$']):
        plt.plot(cut_vals, n, label=label)
    
    plt.legend(fontsize=24)

    plt.xlabel('cut', fontsize=24)
    plt.ylabel('n / year', fontsize=24)

    plt.gca().set_yscale('log')
    
    if not name.endswith('.pdf'):
        name += '.pdf'
    
    plt.savefig(f'{name}', bbox='tight')
    
    plt.clf()
    
    plt.rcParams['figure.figsize'] = orig_size