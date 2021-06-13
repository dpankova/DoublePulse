import numpy as np
from scipy import optimize as sco
from stats import check_random_state

class BinnedPoissonLikelihood():
    '''
    Likelihood: Binned Poisson
    Model: P_B(x) + lmd_S * P_S(x)
    '''
    def __init__(self, sig_hist, bkg_hist, random_state):
        self.sig_hist = sig_hist
        self.bkg_hist = bkg_hist
        self.random_state = check_random_state(random_state)

    def log_likelihood(self, lmd, data_hist):
        expected_hist = (self.bkg_hist + lmd * self.sig_hist)
        result = np.sum(data_hist * np.log(expected_hist) - expected_hist)
        return -result

    def test_statistic(self, lmd, data_hist):
        expected_hist = (self.bkg_hist + lmd * self.sig_hist)
        null_hist = self.bkg_hist
        result = np.sum(data_hist * (np.log(expected_hist) -
                        np.log(null_hist)) - expected_hist + null_hist)
        return -result
    
    def test_statistic_best(self, lmd, true_lmd, data_hist):
        expected_hist = (self.bkg_hist + lmd * self.sig_hist)
        null_hist = (self.bkg_hist + true_lmd * self.sig_hist)
        result = np.sum(data_hist * (np.log(expected_hist) -
                        np.log(null_hist)) - expected_hist + null_hist)
        return -result
    
    def fit_lmd(self, data_hist):
        pars = [0.5]
        par_bounds = [(0, 100)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.test_statistic,
            x0=pars,
            bounds=par_bounds,
            args=(data_hist,),
            approx_grad=True
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or b'FACTR' in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = self.random_state.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.test_statistic,
                x0=pars,
                bounds=par_bounds,
                args=(data_hist,),
                approx_grad=True
            )
            i += 1

        return fmin, xmin
    
    def fit_lmd_best(self, true_lmd, data_hist):
        pars = [0.5]
        par_bounds = [(0, 100)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.test_statistic_best,
            x0=pars,
            bounds=par_bounds,
            args=(true_lmd, data_hist,),
            approx_grad=True
        )
        #print(xmin,fmin,min_dict)

        #print("min_dict1",min_dict)
        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or b'FACTR' in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = self.random_state.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.test_statistic_best,
                x0=pars,
                bounds=par_bounds,
                args=(true_lmd, data_hist,),
                approx_grad=True
            )
            #print(xmin,fmin,min_dict)
            #print("min_dict",min_dict)
            i += 1

        return fmin, xmin
    
class HistLikelihood(object):
    '''Likelihood: (P_B(x) + lmd P_S(x)) / (1 + lmd)
    '''
    def __init__(self, sig_pdf, bkg_pdf, random_state):
        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf
        self.random_state = check_random_state(random_state)

    def log_likelihood(self, lmd, hist_idx):
        nominator = self.bkg_pdf[hist_idx] + lmd * self.sig_pdf[hist_idx]
        result = -np.sum(np.log(nominator) - np.log1p(lmd))
        result += np.sum(np.log(self.bkg_pdf[hist_idx]))

        grad = len(hist_idx) / (1 + lmd) - \
            (np.sum(self.sig_pdf[hist_idx] / nominator))

        # Multiply by 2 for a chi2-distributed test statistic
        result *= 2
        grad *= 2
        return result, grad

    def fit_lmd(self, hist_idx):
        pars = [0.5]
        par_bounds = [(0, 1)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.log_likelihood,
            x0=pars,
            bounds=par_bounds,
            args=hist_idx
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or b'FACTR' in min_dict["task"]:
            if i > 100:
               # print("Did not manage good fit")
                #print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = self.random_state.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.log_likelihood,
                x0=pars,
                bounds=par_bounds,
                args=hist_idx)
            i += 1

        return fmin, xmin

class OneBin_BinnedPoissonLikelihood():
    '''
    Likelihood: Binned Poisson for a single bin case
    Model: P_B(x) + lmd_S * P_S(x)
    '''
    def __init__(self, sig, bkg, random_state):
        self.sig = sig
        self.bkg = bkg
        self.random_state = random_state

    def log_likelihood(self, lmd, data):
        expected = self.bkg + lmd * self.sig
        result =  data * np.log(expected) - expected
        return -result

    def test_statistic(self, lmd, data):
        expected = self.bkg + lmd * self.sig
        null = self.bkg
        result = data * (np.log(expected) - np.log(null)) - expected + null
        return -result
    
    def test_statistic_best(self, lmd, true_lmd, data):
        expected = self.bkg + lmd * self.sig
        null = self.bkg + true_lmd * self.sig
        result = data * (np.log(expected) - np.log(null)) - expected + null
        return -result
    
    def fit_lmd(self, data):
        pars = [0.5]
        par_bounds = [(0, 100)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.test_statistic,
            x0=pars,
            bounds=par_bounds,
            args=(data,),
            approx_grad=True
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or b'FACTR' in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = self.random_state.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.test_statistic,
                x0=pars,
                bounds=par_bounds,
                args=(data,),
                approx_grad=True
            )
            i += 1

        return fmin, xmin
    
    def fit_lmd_best(self, true_lmd, data):
        pars = [0.5]
        par_bounds = [(0, 100)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.test_statistic_best,
            x0=pars,
            bounds=par_bounds,
            args=(true_lmd, data,),
            approx_grad=True
        )
        i = 1
        while min_dict["warnflag"] == 2 or b'FACTR' in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = self.random_state.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.test_statistic_best,
                x0=pars,
                bounds=par_bounds,
                args=(true_lmd, data,),
                approx_grad=True
            )
            i += 1

        return fmin, xmin