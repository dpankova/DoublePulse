from __future__ import division, print_function
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from helper_functions import load_neyman_plane
from helper_functions import make_hists_and_exps
from signal_injector import SignalInjector,SignalInjectorOneBin

def get_acc_vals_from_ts_vals(tss, alpha=0.9):
    acc_vals = np.zeros(tss.shape[0])
    n_samples = tss.shape[1]
    for i, row in enumerate(tss):
        critical_val = np.quantile(row, alpha)
        acc_vals[i] = critical_val

    return acc_vals


def calc_limits_from_trials(likelihood, bkg_samples, mus, acceptance_values, signal_expectation):
    lower_limits = []
    upper_limits = []

    for i in tqdm(range(len(bkg_samples))):
        accepted_mus = np.zeros_like(mus, dtype=int)
        ts, lmd_bf = likelihood.fit_lmd(bkg_samples[i])
        for j, mu in enumerate(mus):
            lmd = mu / signal_expectation
            bf_ts = likelihood.test_statistic_best(lmd_bf, lmd, bkg_samples[i])

            if acceptance_values[j] >= (-bf_ts):
                accepted_mus[j] = 1

        accepted_mu_mask = np.array(accepted_mus, dtype=bool)
        lower_limit = np.min(mus[accepted_mu_mask])
        upper_limit = np.max(mus[accepted_mu_mask])
        lower_limits.append(lower_limit)
        upper_limits.append(upper_limit)
    return np.array(lower_limits), np.array(upper_limits)


class LimitCalculator(object):
    def __init__(self, name, flux_systematic,
                 scan_path, df_path=None,
                 livetime=(3600 * 24 * 365 * 10),
                 random_state=1234,
                 alpha=0.9):
        self.name = name
        self.scan_path = scan_path
        self.flux_systematic = flux_systematic
        self.random_state = np.random.RandomState(random_state)
        self.livetime = livetime
        self.alpha = alpha

        if flux_systematic is True:
            #self.df_path = os.path.join(
            #    '/net/nfshome/home/mmeier/level5/loading/',
            #    'cached_dfs_aachen8yr_muongun_add_numu.hd5')
            if df_path is None:
                self.df_path = os.path.join(
                    '/net/nfshome/home/mmeier/level5/loading/',
                    'cached_dfs_aachen8yr_livetime.hd5')
            else:
                self.df_path = df_path
        else:
            self.df_path = os.path.join(
                '/home/dup193/work/double_pulse/max_test/diff_lim')
                #'cached_{}.hd5'.format(self.name))

        self.__load_acceptance_values__()
        self.__load_hists_and_exps__()

    def __load_acceptance_values__(self, return_ts=False):
        mus, tss = load_neyman_plane(self.scan_path)
        self.mus = mus
        self.acc_vals = get_acc_vals_from_ts_vals(tss, alpha=self.alpha)
        if return_ts:
            return tss
        else:
            del tss

    def __load_hists_and_exps__(self):
        if self.flux_systematic:
            if self.name.lower() == 'baseline':
                weight_col = 'weight'
            else:
                splitted_sys = np.append(['Weight_'], self.name.split('_'))
                weight_col = ''.join([split_i.capitalize() for split_i in splitted_sys])
        else:
            weight_col = 'weight'

        sig, bkg, sig_exp, bkg_exp = make_hists_and_exps(
            self.df_path, weight_col, self.livetime)
        self.sig, self.bkg = sig, bkg
        self.sig_exp, self.bkg_exp = sig_exp, bkg_exp

    def set_likelihood(self, likelihood):
        self.likelihood = likelihood

    def create_samples(self, n_samples=1000, bkg_only=True, sig_inj=None):
        if bkg_only and sig_inj is not None:
            raise ValueError('Samples can be either background only or ' +
                             'have a fixed amount of signal events injected!')
        if bkg_only:
            sig_inj = SignalInjector(self.sig, self.bkg,
                                     random_state=self.random_state,
                                     sig_exp=0.)
        elif sig_inj is not None:
            sig_inj = SignalInjector(self.sig, self.bkg,
                                     random_state=self.random_state,
                                     sig_exp=sig_inj)
        else:
            raise ValueError('Either use the bkg_only option or supply ' +
                             'an expected amount of signal!')

        self.samples = sig_inj.return_samples(n_samples)
        self.injected_signal = sig_inj.sig_exp
        return self.samples

    def calculate_limits(self, acc_vals, flux_norm, samples=None):
        if samples is None:
            samples = self.samples
        sig_exp = np.sum(self.likelihood.sig_hist)
        self.lower_limits, self.upper_limits = calc_limits_from_trials(
            self.likelihood, samples, self.mus, acc_vals, sig_exp)

        self.lower_limits_flux = self.lower_limits / sig_exp * flux_norm
        self.upper_limits_flux = self.upper_limits / sig_exp * flux_norm

    def check_coverage(self):
        n_samples_covered = np.sum(np.logical_and(
            self.lower_limits <= self.injected_signal,
            self.upper_limits >= self.injected_signal))
        n_samples_total = len(self.lower_limits)
        return n_samples_covered / n_samples_total

    @property
    def average_upper_limit(self):
        return np.mean(self.upper_limits_flux)

    @property
    def average_lower_limit(self):
        return np.mean(self.lower_limits_flux)
    
    
class LimitCalculatorNoDF(object):
    def __init__(self, name, flux_systematic,
                 scan_path, sig, bkg,
                 livetime=(3600 * 24 * 365 * 10),
                 random_state=1234,
                 alpha=0.9):
        self.name = name
        self.scan_path = scan_path
        self.flux_systematic = flux_systematic
        self.random_state = np.random.RandomState(random_state)
        self.livetime = livetime
        self.alpha = alpha

        self.sig, self.bkg = sig, bkg
        self.sig_exp, self.bkg_exp = np.sum(sig), np.sum(bkg)

        if flux_systematic is True:
            #self.df_path = os.path.join(
            #    '/net/nfshome/home/mmeier/level5/loading/',
            #    'cached_dfs_aachen8yr_muongun_add_numu.hd5')
            if df_path is None:
                self.df_path = os.path.join(
                    '/net/nfshome/home/mmeier/level5/loading/',
                    'cached_dfs_aachen8yr_livetime.hd5')
            else:
                self.df_path = df_path
        else:
            self.df_path = os.path.join(
                '/home/dup193/work/double_pulse/max_test/diff_lim')
                #'cached_{}.hd5'.format(self.name))

        self.__load_acceptance_values__()
     
    def __load_acceptance_values__(self, return_ts=False):
        mus, tss = load_neyman_plane(self.scan_path)
        self.mus = mus
        self.acc_vals = get_acc_vals_from_ts_vals(tss, alpha=self.alpha)
        if return_ts:
            return tss
        else:
            del tss

    def set_likelihood(self, likelihood):
        self.likelihood = likelihood

    def create_samples(self, n_samples=1000, bkg_only=True, onebin = False, sig_inj=None):
        if onebin:
            sig_inj = SignalInjectorOneBin(self.sig, self.bkg,
                                     random_state=self.random_state,
                                     sig_exp=0.)
        else:    
            if bkg_only and sig_inj is not None:
                raise ValueError('Samples can be either background only or ' +
                                 'have a fixed amount of signal events injected!')
            if bkg_only:
                sig_inj = SignalInjector(self.sig, self.bkg,
                                         random_state=self.random_state,
                                         sig_exp=0.)
            elif sig_inj is not None:
                sig_inj = SignalInjector(self.sig, self.bkg,
                                         random_state=self.random_state,
                                         sig_exp=sig_inj)
            else:
                raise ValueError('Either use the bkg_only option or supply ' +
                                 'an expected amount of signal!')

        self.samples = sig_inj.return_samples(n_samples)
        self.injected_signal = sig_inj.sig_exp
        return self.samples

    def calculate_limits(self, acc_vals, flux_norm, samples=None):
        if samples is None:
            samples = self.samples
        sig_exp = np.sum(self.likelihood.sig_hist)
        self.lower_limits, self.upper_limits = calc_limits_from_trials(
            self.likelihood, samples, self.mus, acc_vals, sig_exp)

        self.lower_limits_flux = self.lower_limits / sig_exp * flux_norm
        self.upper_limits_flux = self.upper_limits / sig_exp * flux_norm

    def check_coverage(self):
        n_samples_covered = np.sum(np.logical_and(
            self.lower_limits <= self.injected_signal,
            self.upper_limits >= self.injected_signal))
        n_samples_total = len(self.lower_limits)
        return n_samples_covered / n_samples_total

    @property
    def average_upper_limit(self):
        return np.mean(self.upper_limits_flux)

    @property
    def average_lower_limit(self):
        return np.mean(self.lower_limits_flux)