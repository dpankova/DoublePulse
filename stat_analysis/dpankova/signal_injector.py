from __future__ import division, print_function
import numpy as np
from stats import sample_from_hist
from stats import check_random_state
from tqdm import tqdm

class SignalInjector(object):
    def __init__(self,
                 sig_dist, bkg_dist,
                 random_state,
                 sig_exp=None, bkg_exp=None):
        if sig_exp is not None:
            self.sig_exp = sig_exp
        else:
            self.sig_exp = np.sum(sig_dist)
        if bkg_exp is not None:
            self.bkg_exp = bkg_exp
        else:
            self.bkg_exp = np.sum(bkg_dist)

        self.sig_dist = sig_dist
        self.bkg_dist = bkg_dist
        self.random_state = check_random_state(random_state)

    def calc_n_events_per_trial(self, n_trials):
        n_sig_per_trial = self.random_state.poisson(lam=self.sig_exp, size=n_trials)
        n_bkg_per_trial = self.random_state.poisson(lam=self.bkg_exp, size=n_trials)
        return n_sig_per_trial, n_bkg_per_trial

    def create_samples(self):
        dists = [self.sig_dist, self.bkg_dist]
        n_samples = [self.n_sig_per_trial,
                     self.n_bkg_per_trial]
        samples_sig, samples_bkg = [], []
        for i, dist in enumerate(dists):
            for j in range(len(n_samples[i])):
                sample = sample_from_hist(dist, n_samples[i][j], self.random_state)
                if i == 0:
                    samples_sig.append(sample)
                elif i == 1:
                    samples_bkg.append(sample)
        self.sig_samples = samples_sig
        self.bkg_samples = samples_bkg
        return samples_sig, samples_bkg
    
    def return_samples(self, n_trials):
        self.n_sig_per_trial, self.n_bkg_per_trial = \
            self.calc_n_events_per_trial(n_trials)
        
        self.create_samples()
        return np.array(self.sig_samples) + np.array(self.bkg_samples)

    def do_trials(self, n_trials, llh):
        self.n_sig_per_trial, self.n_bkg_per_trial = \
            self.calc_n_events_per_trial(n_trials)

        self.create_samples()

        ts_vals = []
        lmds = []
        
        true_lmd = self.sig_exp / float(np.sum(self.sig_dist))
        #print("true lmd", true_lmd,self.sig_exp,float(np.sum(self.sig_dist)))
        for i in range(n_trials):
        #for i in tqdm(range(n_trials)):
            samples = (self.sig_samples[i] + self.bkg_samples[i])

            ts, lmd = llh.fit_lmd_best(true_lmd, samples)
            ts_vals.append(-ts)
            lmds.extend(lmd)
        # print(np.mean(lmds), np.median(lmds))
        return ts_vals, lmds
    
class SignalInjectorOneBin(object):
    def __init__(self, sig, bkg, random_state, sig_exp=None, bkg_exp=None):
        if sig_exp is not None:
            self.sig_exp = sig_exp
        else:
            self.sig = sig
            
        if bkg_exp is not None:
            self.bkg_exp = bkg_exp
        else:
            self.bkg_exp = bkg

        self.sig = sig
        self.bkg = bkg
        self.random_state = random_state

    def calc_n_trials(self, n_trials):
        n_sig_trials = self.random_state.poisson(lam=self.sig_exp, size=n_trials)
        n_bkg_trials = self.random_state.poisson(lam=self.bkg_exp, size=n_trials)
        return n_sig_trials, n_bkg_trials

    def return_samples(self, n_trials):
        self.n_sig_per_trial, self.n_bkg_per_trial = \
            self.calc_n_trials(n_trials)

        return  self.n_sig_per_trial + self.n_bkg_per_trial
    
    def do_trials(self, n_trials, llh):
        self.sig_trials, self.bkg_trials = self.calc_n_trials(n_trials)
        
        ts_vals = []
        lmds = []
        true_lmd = self.sig_exp / float(np.sum(self.sig))
        
        for i in range(n_trials):
            samples = (self.sig_trials[i] + self.bkg_trials[i])
            ts, lmd = llh.fit_lmd_best(true_lmd, samples)
            ts_vals.append(-ts)
            lmds.extend(lmd)
        return ts_vals, lmds