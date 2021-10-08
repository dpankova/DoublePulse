"""
Utilities for the nu_tau CNN statistical analysis 
"""

import nuflux
import numba
import numpy as np

import plotting

DEFAULT_UPPER_BOUND = 50
DEFAULT_STEP_SIZE = 0.1


class Analysis:
    """Encapsulates configurable parameters of the nu_tau flux analysis"""

    def __init__(
        self,
        exp_df,
        livetime,
        net1_bins,
        net3_bins,
        astro_phi,
        astro_gamma,
        conv_model,
        prompt_model,
        syst_slopes,
        syst_widths,
        prior_types,
        astro_norm_mean,
        astro_norm_width,
        conv_norm_mean,
        conv_norm_width,
        prompt_norm_mean,
        prompt_norm_width,
        mg_norm_mean,
        mg_norm_width,
        rng=None,
    ):
        """Initialize from a DataFrame and configuration parameters
        """
        conv_flux = nuflux.makeFlux(conv_model).getFlux
        weight_conv = lambda d: calc_atmos_weights(d, conv_flux) * livetime
        prompt_flux = nuflux.makeFlux(prompt_model).getFlux
        weight_prompt = lambda d: calc_atmos_weights(d, prompt_flux) * livetime
        weight_astro = (
            lambda d: calc_astro_weights(d, astro_phi, astro_gamma) * livetime
        )
        # muon gun weights are precalculated and stored in the "oneweight" field
        weight_mg = lambda d: d.oneweight * livetime

        tau_CC_mask = (np.abs(exp_df.pid) == 16) & (exp_df.it == 1)
        # pid == 0 indicates muon gun
        mg_mask = exp_df.pid == 0
        nu_mask = ~mg_mask

        self._group_names = [
            r"astrophysical $\nu_\tau$ CC",
            "astrophysical bg",
            "conventional atmos bg",
            "prompt atmos bg",
            "cosmic muon bg",
        ]
        group_weight_funcs = [
            weight_astro,
            weight_astro,
            weight_conv,
            weight_prompt,
            weight_mg,
        ]
        self._group_nominals = [
            1.0,
            astro_norm_mean,
            conv_norm_mean,
            prompt_norm_mean,
            mg_norm_mean,
        ]
        self._group_scales = [
            0.0,
            astro_norm_width,
            conv_norm_width,
            prompt_norm_width,
            mg_norm_width,
        ]
        group_masks = [tau_CC_mask, nu_mask & (~tau_CC_mask), nu_mask, nu_mask, mg_mask]
        self._group_hists = [
            build_template(exp_df[mask], net1_bins, net3_bins, "n1", "n3", weight_f)
            for mask, weight_f in zip(group_masks, group_weight_funcs)
        ]
        # note: we're using the same MC events for astro and atmospheric bg estimations,
        # so the MC uncertainties are correlated between different the various bg templates
        self._group_vars = [
            build_template(
                exp_df[mask],
                net1_bins,
                net3_bins,
                "n1",
                "n3",
                lambda d: weight_f(d) ** 2,
            )
            for mask, weight_f in zip(group_masks, group_weight_funcs)
        ]

        self._sig_hist = self._group_hists[0]
        self._bg_hist = sum(self._group_hists[1:])
        self._n1_bins = net1_bins
        self._n3_bins = net3_bins

        n_syst_params = len(syst_slopes)
        if len(syst_widths) != n_syst_params or len(prior_types) != n_syst_params:
            raise ValueError(
                "`syst_slopes`, `syst_widths`, and `prior_types` must all be the same length."
            )

        self._syst_slopes = np.array(syst_slopes)
        self._syst_widths = np.array(syst_widths)
        self.validate_prior_types(prior_types)
        self._prior_types = prior_types

        self._rng = np.random.default_rng() if rng is None else rng

    @property
    def sig_hist(self):
        return self._sig_hist

    @property
    def bg_hist(self):
        return self._bg_hist

    @property
    def group_hists(self):
        return self._group_hists

    @property
    def group_nominals(self):
        return self._group_nominals

    @property
    def group_scales(self):
        return self._group_scales

    @property
    def group_vars(self):
        return self._group_vars

    @property
    def group_names(self):
        return self._group_names

    def sample(self, lambda_tau=1, n_hists=None, with_syst=False):
        """generate `n_hists` pseudoexperiment histograms"""
        n_hists = 1 if n_hists is None else n_hists

        if with_syst:
            # randomize fluxes but not the tau flux; that is fixed to lambda_tau
            nominals, scales = self._group_nominals[1:], self._group_scales[1:]
            return sample_with_syst(
                lambda_tau,
                n_hists,
                self.group_hists,
                nominals,
                scales,
                self._syst_slopes,
                self._syst_widths,
                self._prior_types,
                self._rng,
            )
        else:
            # sample without flux uncertainties or MC uncertainties
            return sample(
                lambda_tau, self._rng, self.sig_hist, self.bg_hist, n_hists
            ).squeeze()

    def calculate_TS(self, lambda_tau, obs_hists):
        """calculate the TS for each hist in `hists`"""
        return calculate_TS(lambda_tau, obs_hists, self.sig_hist, self.bg_hist)

    def sample_TS(self, lambda_tau, n_trials=10000, with_syst=False):
        """obtain a sample of TS values for `lambda_tau` using `n_trials` pseudo experiments"""
        hists = self.sample(lambda_tau, n_trials, with_syst)
        return self.calculate_TS(lambda_tau, hists)

    def scan_TS(self, obs_hist, lambda_taus):
        """calculate the TS for each lambda_tau in `lambda_taus`"""
        return scan_TS(obs_hist, lambda_taus, self.sig_hist, self.bg_hist)

    def fit_lambda_tau(
        self, obs_hist, upper_bound=DEFAULT_UPPER_BOUND, step_size=DEFAULT_STEP_SIZE
    ):
        """find the best fit lambda_tau for the given observation and templates"""
        return fit_lambda_tau(obs_hist, self.sig_hist, self.bg_hist, upper_bound)

    def llh_scan(self, obs_hist, lambda_taus):
        """calculate the Poisson LLH for each lambda in `lambdas`"""
        return llh_scan(obs_hist, lambda_taus, self.sig_hist, self.bg_hist)

    def build_TS_plane(
        self, max_lambda, n_steps, trials_per_lambda, conf_levels, with_syst=False
    ):
        """sample TS, calculate critical values """
        ts_plane = self.get_TS_samples(
            max_lambda, n_steps, trials_per_lambda, with_syst
        )

        return self.calculate_critical_values(ts_plane, conf_levels)

    def get_TS_samples(self, max_lambda, n_steps, trials_per_lambda, with_syst=False):
        """sample TS for `n_steps` lambdas in the range 0, `max_lambda`]"""
        lambda_taus = np.linspace(0, max_lambda, n_steps)
        ts_samples = np.empty((n_steps, trials_per_lambda))

        for i, l in enumerate(lambda_taus):
            ts_samples[i] = self.sample_TS(l, trials_per_lambda, with_syst)[0]

        return dict(lambdas=lambda_taus, ts_samples=ts_samples)

    @staticmethod
    def calculate_critical_values(ts_plane, conf_levels):
        """Calculates critical values and adds them to the ts_plane dict"""
        critical_values = np.empty((len(ts_plane["lambdas"]), len(conf_levels)))
        for i, samps in enumerate(ts_plane["ts_samples"]):
            for j, cl in enumerate(conf_levels):
                critical_values[i][j] = np.percentile(
                    samps, 100 * cl, interpolation="higher"
                )

        ts_plane.update(dict(critical_values=critical_values, conf_levels=conf_levels))

        return ts_plane

    def plot_templates(self):
        plotting.plot_hists(self._n1_bins, self._n3_bins, self.sig_hist, self.bg_hist)

    def plot_sample(self, ax, lambda_tau=1, n_obs=None):
        while True:
            sample_hist = self.sample(lambda_tau)
            if n_obs is None or sample_hist.sum() == n_obs:
                break

        plotting.plot_hist(self._n1_bins, self._n3_bins, sample_hist, ax)
        ax.set_title(rf"Pseudo experiment, $\lambda_\tau$ = {lambda_tau:.1f}")

        return sample_hist

    def plot_groups(self, axes):
        """plot the group means and vars"""
        axiter = axes.flat
        for mean_hist, var_hist, label in zip(
            self.group_hists, self.group_vars, self.group_names
        ):
            ax = next(axiter)
            plotting.plot_hist(self._n1_bins, self._n3_bins, mean_hist, ax)
            ax.set_title(f"{label} nominal counts")

            ax = next(axiter)
            mean_hist = np.where(mean_hist == 0, 1, mean_hist)
            rel_unc = np.sqrt(var_hist) / mean_hist
            plotting.plot_hist(self._n1_bins, self._n3_bins, rel_unc, ax)
            ax.set_title(f"{label} relative MC uncertainty")

    @staticmethod
    def validate_prior_types(prior_types, valid_types=("norm", "uniform")):
        for prior in prior_types:
            if prior not in valid_types:
                raise ValueError(
                    f"{prior} is an unrecognized prior type. Must be one of {valid_types}"
                )


def calc_astro_weights(astro_d, phi, gamma):
    en_weights = 1e-18 * phi * (astro_d.energy / 1e5) ** (-gamma)
    return en_weights * astro_d.oneweight / astro_d.n_files / astro_d.n_events


def calc_atmos_weights(atmos_d, flux):
    flux_weight = flux(atmos_d.pid.astype(np.int32), atmos_d.energy, atmos_d.coszen)

    denom = atmos_d.n_files * atmos_d.n_events * atmos_d.typeweight
    return flux_weight * atmos_d.oneweight / denom


def build_template(df, x_bins, y_bins, x_key, y_key, weight_func):
    """Create a 2d histogram from a DataFrame"""
    return np.histogram2d(
        df[x_key], df[y_key], (x_bins, y_bins), weights=weight_func(df)
    )[0]


#
# Analysis method implementations
#


def sample(lambda_tau, rng, sig_hist, bg_hist, n_hists=None):
    if n_hists is not None:
        size = (n_hists,) + sig_hist.shape

    exp = lambda_tau * sig_hist + bg_hist
    return rng.poisson(exp, size)


@numba.njit
def poisson_llh(obs, lambda_tau, sig_hist, bg_hist):
    """calculate a Poisson LLH"""
    exp = lambda_tau * sig_hist + bg_hist

    return (obs * np.log(exp) - exp).sum()


@numba.njit
def llh_scan(obs, lambdas, sig_hist, bg_hist):
    return np.array([poisson_llh(obs, l, sig_hist, bg_hist) for l in lambdas])


@numba.njit
def fit_lambda_tau(
    obs, sig_hist, bg_hist, upper_bound=DEFAULT_UPPER_BOUND, step_size=DEFAULT_STEP_SIZE
):
    """ fit_lambda_tau implementation

    uses a binary search. Fit results are restricted to be positive
    """
    left = 0
    right = upper_bound

    # binary search to find lambda_tau maximizing the LLH
    while left < right - step_size:
        mid = (left + right) / 2
        llh_mid = poisson_llh(obs, mid, sig_hist, bg_hist)
        llh_right = poisson_llh(obs, mid + step_size, sig_hist, bg_hist)
        if llh_right >= llh_mid:
            left = mid + step_size
        else:
            right = mid

    if left == 0:
        scan_pts = np.linspace(0, step_size, 11)
        scan_llhs = llh_scan(obs, scan_pts, sig_hist, bg_hist)
        argmax = np.argmax(scan_llhs)
        if argmax == 0:
            return 0, scan_llhs[0]
        else:
            # run interpolation with a smaller step size
            left = scan_pts[argmax]
            step_size = step_size / 10

    # interpolate the best value
    llh_left = poisson_llh(obs, left - step_size, sig_hist, bg_hist)
    llh_mid = poisson_llh(obs, left, sig_hist, bg_hist)
    llh_right = poisson_llh(obs, left + step_size, sig_hist, bg_hist)

    delta_l = llh_left - llh_mid
    delta_r = llh_right - llh_mid
    tau_min = left + step_size * ((delta_l - delta_r) / (2 * (delta_l + delta_r)))

    return tau_min, poisson_llh(obs, tau_min, sig_hist, bg_hist)


@numba.njit
def calculate_TS(lambda_tau, hists, sig_hist, bg_hist):
    n_hists = hists.shape[0]
    ts = np.empty(n_hists)
    best_fits = np.empty(n_hists)
    for i in range(n_hists):
        obs = hists[i]
        best_fit, best_llh = fit_lambda_tau(obs, sig_hist, bg_hist)
        best_fits[i] = best_fit
        ts[i] = best_llh - poisson_llh(obs, lambda_tau, sig_hist, bg_hist)

    return ts, best_fits


@numba.njit
def scan_TS(obs, lambda_taus, sig_hist, bg_hist):
    out = np.empty_like(lambda_taus)
    best_fit, best_llh = fit_lambda_tau(obs, sig_hist, bg_hist)
    for i, l in enumerate(lambda_taus):
        out[i] = best_llh - poisson_llh(obs, l, sig_hist, bg_hist)

    return out, best_fit, best_llh


def combine_comps(comp_means, scaling_factors):
    # combined_means = sum(k * means for means, k in zip(comp_means, scaling_factors))
    # combined_vars = sum(
    #     k ** 2 * variance for variance, k in zip(comp_vars, scaling_factors)
    # )
    return sum(k * means for means, k in zip(comp_means, scaling_factors))


def sample_fluxes(nominals, scales, n_trials, rng):
    epsilons = rng.normal(size=(len(nominals), n_trials, 1, 1))

    return tuple(
        np.clip(nom + scale * eps, a_min=0, a_max=None)
        for nom, scale, eps in zip(nominals, scales, epsilons)
    )


def sample_detector_systs(syst_slopes, syst_widths, prior_types, n_trials, rng):
    epsilons = np.empty(shape=(n_trials, len(prior_types)))
    for i, ptype in enumerate(prior_types):
        if ptype == "norm":
            epsilons[:, i] = rng.normal(size=n_trials)
        elif ptype == "uniform":
            epsilons[:, i] = rng.uniform(low=-1.0, high=1.0, size=n_trials)
        else:
            raise ValueError(f"{ptype} is an unrecognized prior type!")

    scales = np.clip(
        1 + (syst_slopes * syst_widths * epsilons).sum(axis=1), a_min=0, a_max=None
    )
    scales.shape = (n_trials, 1, 1)
    return scales


def sample_with_syst(
    lambda_tau,
    n_trials,
    group_hists,
    nominals,
    scales,
    syst_slopes,
    syst_widths,
    prior_types,
    rng,
):
    # first generate the hist means and uncertainties for each trial
    detector_syst_scales = sample_detector_systs(
        syst_slopes, syst_widths, prior_types, n_trials, rng
    )
    tau_flux_scales = lambda_tau * np.ones((n_trials, 1, 1))
    flux_scales = (tau_flux_scales,) + sample_fluxes(nominals, scales, n_trials, rng)
    scaling_factors = tuple(detector_syst_scales * flux for flux in flux_scales)

    lambdas = combine_comps(group_hists, scaling_factors)

    # ATF note: removed flucatuation according to MC uncertainties;
    # it complicates things and has no appreciate effect on the results
    # perturb lambdas randomly according to MC uncertainties
    # eps = rng.normal(size=(means.shape))

    # lambdas = np.clip(means + stds * eps, a_min=0, a_max=None)
    return rng.poisson(lambdas)
