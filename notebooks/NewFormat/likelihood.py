from scipy import optimize as sco

def check_random_state(random_state):
    if isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        if not isinstance(random_state, int):
            raise ValueError('random_state has to be either an int or of ' +
                             'type np.random.RandomState!')
        else:
            random_state = np.random.RandomState(random_state)
            return random_state


class Likelihood(object):
    '''Likelihood: (P_B(x) + lmd P_S(x)) / (1 + lmd)
    '''
    def __init__(self, sig_pdf, bkg_pdf):
        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf

    def log_likelihood(self, lmd, scores):
        nominator = self.bkg_pdf(scores) + lmd * self.sig_pdf(scores)
        result = -np.sum(np.log(nominator) - np.log1p(lmd))
        result += np.sum(np.log(self.bkg_pdf(scores)))

        grad = len(scores) / (1 + lmd) - \
            (np.sum(self.sig_pdf(scores) / nominator))

        # Multiply by 2 for a chi2-distributed test statistic
        result *= 2
        grad *= 2
        return result, grad

    def fit_lmd(self, scores):
        pars = [0.5]
        par_bounds = [(0, 1)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.log_likelihood,
            x0=pars,
            bounds=par_bounds,
            args=scores
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
            if i > 100:
                # print(np.min(scores), np.max(scores))
                # print(np.sum(~np.isfinite(scores)))
                # llh_scan = np.zeros(101)
                # grad_scan = np.zeros(101)
                # for i in range(len(llh_scan)):
                #     lmdi = i / 100.
                #     llh_scan[i], grad_scan[i] = self.log_likelihood(
                #     lmdi, scores)
                # print(llh_scan)
                # print(grad_scan)
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = np.random.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.log_likelihood,
                x0=pars,
                bounds=par_bounds,
                args=scores)
            i += 1

        return fmin, xmin


class LikelihoodBoth(Likelihood):
    '''Likelihood: (1 - lmd) P_B(x) + lmd P_S(x)
    '''
    def log_likelihood(self, lmd, scores):
        nominator = (1 - lmd) * self.bkg_pdf(scores) + \
            lmd * self.sig_pdf(scores)
        result = np.sum(np.log(nominator))
        result -= np.sum(np.log(self.bkg_pdf(scores)))

        grad = -(self.sig_pdf(scores) - self.bkg_pdf(scores)) / nominator

        # Multiply by 2 for a chi2-distributed test statistic
        result *= -2
        grad *= 2
        return result, grad


class ExtendedLikelihood(object):
    '''Likelihood: exp(-(lmd_S + lmd_B)) prod_i (lmd_B P_B(x) + lmd_S P_S(x))
    '''
    def __init__(self, sig_pdf, bkg_pdf):
        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf

    def log_likelihood(self, lmds, scores):
        lmd_s, lmd_b = lmds
        N = len(scores)
        nominator = lmd_b * self.bkg_pdf(scores) + lmd_s * self.sig_pdf(scores)
        result = - (lmd_s + lmd_b)
        result += np.sum(np.log(nominator))

        result += N
        result -= np.sum(np.log(N * self.bkg_pdf(scores)))

        grad_s = -(-1 + np.sum(self.sig_pdf(scores) / nominator))
        grad_b = -(-1 + np.sum(self.bkg_pdf(scores) / nominator))

        # Multiply by 2 for a chi2-distributed test statistic
        result *= -2
        grad_s *= 2
        grad_b *= 2
        return result, np.array([grad_s, grad_b])

    def fit_lmd(self, scores):
        pars = [0.5, 0.5]
        par_bounds = [(0, 10), (0, 10)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.log_likelihood,
            x0=pars,
            bounds=par_bounds,
            args=scores
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
            if i > 100:
                # print(np.min(scores), np.max(scores))
                # print(np.sum(~np.isfinite(scores)))
                # llh_scan = np.zeros(101)
                # grad_scan = np.zeros(101)
                # for i in range(len(llh_scan)):
                #     lmdi = i / 100.
                #     llh_scan[i], grad_scan[i] = self.log_likelihood(
                #     lmdi, scores)
                # print(llh_scan)
                # print(grad_scan)
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = np.random.uniform(0., 1.)
            pars[1] = np.random.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.log_likelihood,
                x0=pars,
                bounds=par_bounds,
                args=scores)
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
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
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


class ExtendedHistLikelihood(object):
    '''Likelihood: exp(-(lmd_S + lmd_B)) prod_i (lmd_B P_B(x) + lmd_S P_S(x))
    '''
    def __init__(self, sig_pdf, bkg_pdf):
        self.sig_pdf = sig_pdf
        self.bkg_pdf = bkg_pdf

    def log_likelihood(self, lmds, hist_idxs):
        lmd_s, lmd_b = lmds
        if self.sig_pdf.ndim == 2:
            N = len(hist_idxs[0])
        elif self.sig_pdf.ndim == 1:
            N = len(hist_idxs)
        nominator = (lmd_b * self.bkg_pdf[hist_idxs] +
                     lmd_s * self.sig_pdf[hist_idxs])
        result = - (lmd_s + lmd_b)
        result += np.sum(np.log(nominator))

        result += N
        result -= np.sum(np.log(N * self.bkg_pdf[hist_idxs]))

        grad_s = -(-1 + np.sum(self.sig_pdf[hist_idxs] / nominator))
        grad_b = -(-1 + np.sum(self.bkg_pdf[hist_idxs] / nominator))

        # Multiply by 2 for a chi2-distributed test statistic
        result *= -2
        grad_s *= 2
        grad_b *= 2
        return result, np.array([grad_s, grad_b])

    def fit_lmd(self, hist_idxs):
        pars = [0.5, 0.5]
        par_bounds = [(0, 1000), (0, 1000)]

        xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
            func=self.log_likelihood,
            x0=pars,
            bounds=par_bounds,
            args=(hist_idxs,)
        )

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
            if i > 100:
                print("Did not manage good fit")
                print('results are {}, {}'.format(fmin, xmin))
                return fmin, xmin

            pars[0] = np.random.uniform(0., 1.)
            pars[1] = np.random.uniform(0., 1.)

            # no stop due to gradient
            xmin, fmin, min_dict = sco.fmin_l_bfgs_b(
                func=self.log_likelihood,
                x0=pars,
                bounds=par_bounds,
                args=(hist_idxs,))
            i += 1

        return fmin, xmin


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
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
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

        # set up mindict to enter while, exit if fit looks nice
        i = 1
        while min_dict["warnflag"] == 2 or "FACTR" in min_dict["task"]:
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
            i += 1

        return fmin, xmin