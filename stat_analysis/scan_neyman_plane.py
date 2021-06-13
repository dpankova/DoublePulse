from __future__ import division, print_function
import numpy as np
from likelihood import BinnedPoissonLikelihood
from signal_injector import SignalInjector, SignalInjectorOneBin
from concurrent.futures import ProcessPoolExecutor, wait
from glob import glob
import os
import sys

def scan_neyman_row(sig_inj, n_samples, likelihood, mu):
    sig_ts,_ = sig_inj.do_trials(
        n_samples, likelihood)
    sig_ts = np.array(sig_ts)
    sig_ts[sig_ts < 0] = 0
    return sig_ts, mu

def scan_neyman_plane(sig, bkg, bkg_exp,
                      mu_bins, ts_bins, n_samples_per_mu,
                      n_jobs=1, save=False, out_path=None):
    def save_stuff(out_path, hist, mu):
        if out_path is not None:
            out_path1 = os.path.join(
                out_path, 'ts_mu_{}.npz'.format(mu_i))
            np.savez(out_path1, hist)
            #out_path2 = os.path.join(
            #    out_path, 'lmd_mu_{}.npz'.format(mu_i))
            #np.savez(out_path2, lmd)

    if not save:
        hists = np.zeros((len(mu_bins), len(ts_bins) - 1))
        
    random_state = np.random.RandomState(42)
    likelihood = BinnedPoissonLikelihood(sig, bkg, random_state=random_state)

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i, mu_i in enumerate(mu_bins):
                injector_rs = np.random.RandomState(i)
                sig_inj = SignalInjector(sig, bkg, injector_rs, sig_exp=mu_i, bkg_exp=bkg_exp)
                futures.append(
                    executor.submit(
                        scan_neyman_row,
                        sig_inj=sig_inj,
                        n_samples=n_samples_per_mu,
                        likelihood=likelihood,
                        mu=mu_i))
            results = wait(futures)
            
        for i, future_i in enumerate(results.done):
            try:
                sig_ts, mu_i = future_i.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (future_i, exc))
            #idx = np.where(mu_bins == mu_i)[0][0]
            hist, ts_e = np.histogram(sig_ts, bins=ts_bins)
            hist = hist / float(np.sum(hist))
            if not save:
                hists[idx] = hist
            else:
                save_stuff(out_path, sig_ts, mu_i)

    else:
        raise NotImplementedError('Adapt to new implementation!')
        for i, mu_i in tqdm(enumerate(mu_bins)):
            injector_rs = np.random.RandomState(i)
            sig_inj = SignalInjector(sig, bkg, injector_rs, sig_exp=mu_i, bkg_exp=bkg_exp)
            sig_ts, mu_i = scan_neyman_row(sig_inj=sig_inj,n_samples=n_samples_per_mu, likelihood=likelihood,mu=mu_i)
            sig_ts = np.array(sig_ts)
            sig_ts[sig_ts < 0] = 0
            hist, ts_e = np.histogram(sig_ts, bins=ts_bins)
            hist = hist / float(np.sum(hist))
            if not save:
                hists[i] = hist
            else:
                save_stuff(out_path, sig_ts, mu_i)

    if not save:
        return hists, ts_e
    else:
        return 0, 0
    
    
def scan_neyman_plane_OneBin(sig, bkg, bkg_exp, mu_bins, ts_bins, n_samples_per_mu, n_jobs=2, save=False, out_path=None):
    def save_stuff(out_path, hist, mu):
        if out_path is not None:
            out_path1 = os.path.join(out_path, 'ts_mu_{}.npz'.format(mu_i))
            np.savez(out_path1, hist)

    if not save:
        hists = np.zeros((len(mu_bins), len(ts_bins) - 1))
        
    random_state = np.random.RandomState(42)
    likelihood = BinnedPoissonLikelihood(sig, bkg, random_state=random_state)

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i, mu_i in enumerate(mu_bins):
                injector_rs = np.random.RandomState(i)
                sig_inj = SignalInjectorOneBin(sig, bkg, injector_rs, sig_exp=mu_i, bkg_exp=bkg_exp)
                futures.append(
                    executor.submit(
                        scan_neyman_row,
                        sig_inj=sig_inj,
                        n_samples=n_samples_per_mu,
                        likelihood=likelihood,
                        mu=mu_i))
            results = wait(futures)
            
        for i, future_i in enumerate(results.done):
            try:
                sig_ts, mu_i = future_i.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (future_i, exc))
          
            hist, ts_e = np.histogram(sig_ts, bins=ts_bins)
            hist = hist / float(np.sum(hist))
            if not save:
                hists[idx] = hist
            else:
                save_stuff(out_path, sig_ts, mu_i)
                
    if not save:
        return hists, ts_e
    else:
        return 0, 0