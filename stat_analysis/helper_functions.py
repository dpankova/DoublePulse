import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
import os


    
def build_histograms(sig_df, bkg_df, bins_x, bins_y,
                     density=False,
                     weight_col='weight'):
    #axii are reversed in historgamm2d
    sig, xe, ye = np.histogram2d(
        sig_df['n1'],
        sig_df['n3'],
        bins=[bins_x, bins_y],
        weights=sig_df[weight_col],
        density=density)
   
    bkg, xe, ye = np.histogram2d(
        bkg_df['n1'],
        bkg_df['n3'],
        bins=[bins_x, bins_y],
        weights=bkg_df[weight_col],
        density=density)
    return sig, bkg, xe, ye

def build_error_histograms(sig_df, bkg_df, bins_x, bins_y,
                           scaling_factor_s, scaling_factor_b,weight_col='weight'):
    sig_err = np.sqrt(
        np.histogram2d(
            sig_df['n1'],
            sig_df['n3'],
            bins=[bins_x, bins_y],
            weights=sig_df[weight_col]**2,
            density=False)[0]) * scaling_factor_s

    bkg_err = np.sqrt(
        np.histogram2d(
            bkg_df['n1'],
            bkg_df['n3'],
            bins=[bins_x, bins_y],
            weights=bkg_df[weight_col]**2,
            density=False)[0]) * scaling_factor_b

    return sig_err, bkg_err

def plot_histogram(H,xedges,yedges,Name,norm):
    fig = plt.figure(figsize=(5, 5),facecolor ='w')
    plt.rcParams.update({'font.size': 16})
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-5,5))
    ax = fig.add_subplot()
    ax.set_xlabel('NET1 score', fontsize = 16)                                                              
    ax.set_ylabel('NET3 score', fontsize = 16)
    ax.set_title(Name, fontsize = 16)

    X, Y = np.meshgrid(xedges, yedges)
    pc = ax.pcolormesh(X, Y, H/norm)
    fig.colorbar(pc)
    
def get_default_binning():
    bins_x = np.linspace(0.99, 1.0, 5)
    bins_y = np.linspace(0.85, 1.0, 5)
    #bins_x = [0.99,0.995,0.9975,0.99875,0.999375,1]
    #bins_y = [0.85,0.925,0.9625,0.98125,0.990625,1]
    return bins_x, bins_y

def load_neyman_plane(folder='test_statistics'):
    file_names = glob(os.path.join(folder, '*.npz'))
    mus = np.asarray([float(fn.split('_')[-1].replace('.npz', '')) for fn in file_names])
    sort_idx = np.argsort(mus)
    ts_arrs = np.asarray([np.load(fn)['arr_0'] for fn in file_names])
    sorted_mus = mus[sort_idx]
    sorted_ts_arrs = ts_arrs[sort_idx]
    return sorted_mus, sorted_ts_arrs

def make_hists_and_exps(path, weight_col='weight', livetime=1):
    #sig_df, bkg_df, sig_exp, bkg_exp = load_sample_and_combine_with_add_statistics(
    #        from_cache=True,
    #        cached_path=path,
    #        remove_muongun=False)
    sig_df = pd.read_csv(path+'sig_out.csv')
    #sig_exp = np.sum(sig)
    bkg_df = pd.read_csv(path+'bkg_out.csv')
    
    bins_x, bins_y = get_default_binning()

    sig, bkg, xe, ye = build_histograms(
        sig_df, bkg_df, bins_x, bins_y,
        #livetime=livetime, 
        density=False, weight_col=weight_col)

    sig_exp = np.sum(sig)
    bkg_exp = np.sum(bkg)

    del sig_df
    del bkg_df

    return sig, bkg, sig_exp, bkg_exp

def build_uncert_histograms(sig_df, bkg_df, bins_x, bins_y,
                    # livetime=(3600 * 24 * 365 * 7),
                     density=False,
                     weight_col='weight'):
    #axii are reversed in historgamm2d
    sig, xe, ye = np.histogram2d(
        sig_df['n1'],
        sig_df['n3'],
        bins=[bins_x, bins_y],
        weights=np.square(sig_df[weight_col]),
        density=density)
   
    bkg, xe, ye = np.histogram2d(
        bkg_df['n1'],
        bkg_df['n3'],
        bins=[bins_x, bins_y],
        weights=np.square(bkg_df[weight_col]),
        density=density)
    
    sig = np.sqrt(sig)
    bkg = np.sqrt(bkg)
    
    return sig, bkg, xe, ye

def plot_histogram_ratio(H1,H2,xedges,yedges,Name):
    fig = plt.figure(figsize=(5, 5),facecolor ='w')
    plt.rcParams.update({'font.size': 16})
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-5,5))
    ax = fig.add_subplot()
    ax.set_xlabel('NET1 score', fontsize = 16)                                                              
    ax.set_ylabel('NET3 score', fontsize = 16)
    ax.set_title(Name, fontsize = 16)
    H = np.zeros(H1.shape)
    for i in range(len(H1)):
        for j in range(len(H1[0])):
            H[i,j] = H2[i,j]/H1[i,j] 
    X, Y = np.meshgrid(xedges, yedges)
    pc = ax.pcolormesh(X, Y, H)
    fig.colorbar(pc)