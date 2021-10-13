"""
Plotting functions for the nu_tau CNN statistical analysis 
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from scipy import stats


def plot_hist(x_bins, y_bins, hist, ax):
    """Plot a single histogram"""
    m = ax.pcolormesh(x_bins, y_bins, hist.T, rasterized=True)
    plt.colorbar(m, ax=ax)
    ax.set_xlabel("NET1")
    ax.set_ylabel("NET3")


def plot_hists(x_bins, y_bins, sig_hist, bg_hist, labels=("signal", "background")):
    """ Plot signal and background histograms"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3)

    for ax, hist, label in zip(axes, (sig_hist, bg_hist), labels):
        plot_hist(x_bins, y_bins, hist, ax)
        ax.set_title(label)


def plot_ts_plane(
    ax,
    ts_plane,
    hist_bins,
    cl_color_cycle=("r", "b", "orange"),
    other_cvs=None,
    other_prefix=None,
):
    lambda_taus = ts_plane["lambdas"]

    y_edges = np.hstack((lambda_taus, [2 * lambda_taus[-1] - lambda_taus[-2]]))
    plane_hist = np.empty((len(lambda_taus), len(hist_bins) - 1))
    for i, samples in enumerate(ts_plane["ts_samples"]):
        plane_hist[i] = np.histogram(samples, bins=hist_bins)[0]

    ax.pcolormesh(
        hist_bins,
        y_edges,
        plane_hist,
        rasterized=True,
        cmap="viridis",
        norm=colors.LogNorm(vmin=1, vmax=plane_hist.max()),
    )

    # plot asymptotic limits
    chi2 = stats.chi2(1)
    for cl in ts_plane["conf_levels"]:
        asymptotic_limit = chi2.ppf(cl) / 2.0
        ax.axvline(asymptotic_limit, color="black", linestyle="--")

    # plot critical values
    for i, (cv, cl) in enumerate(
        zip(ts_plane["critical_values"].T, ts_plane["conf_levels"])
    ):
        ax.plot(
            cv, lambda_taus, color=cl_color_cycle[i], label=f"{cl*100:.1f}%",
        )

    if other_cvs is not None:
        for i, (cv, cl) in enumerate(zip(other_cvs.T, ts_plane["conf_levels"])):
            ax.plot(
                cv,
                lambda_taus,
                linestyle="--",
                color=cl_color_cycle[i],
                label=f"{other_prefix} {cl*100:.1f}%",
            )

    ax.set_xlabel(r"$\mathrm{LLH}(\mathrm{best~fit}) - \mathrm{LLH}(\lambda_\tau)$")
    ax.set_ylabel(r"$\lambda_\tau$")
    ax.legend()
