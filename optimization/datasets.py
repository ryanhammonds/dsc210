import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


def generate_datasets(
    iterator, linear=True, random_state=10, n_samples=1000, n_features=3, noise=1
):
    """Create regression datasets with linear or non-linear combinations of random features with noise.

    Nonlinear combinations are created according the the "Friedman #1" formula.

    Parameters
    ----------
    iterator : str
        The argument (as a list) to iterate over. Must be one of `n_samples`, `n_features`, `noise`.
    linear : bool, optional
        Whether to use a linear or non-linear combination of features to produce output, by default True
    random_state : int, optional
        The random state to use, by default 10
    n_samples : int or list, optional
        The number of samples, by default 1000
    n_features : int or list, optional
        The number of features, by default 3 (default 5 for non-linear)
    noise : int or list, optional
        The standard deviation of the gaussian noise applied to the output., by default 1

    Returns
    -------
    Xs: dictionary of ndarrays (n_samples, n_features)
        The input samples for each value of the iterator.
    ys: dictionary of ndarrays (n_samples,)
        The output samples for each value of the iterator.
    """

    kwargs = {"n_samples": n_samples, "n_features": n_features, "noise": noise}
    assert iterator in kwargs.keys(), f"iterator must be one of {kwargs.keys()}"

    Xs = {}
    ys = {}
    coefs = {}
    for key in kwargs[iterator]:
        kwargs[iterator] = key
        if linear:
            X, y, coef = datasets.make_regression(
                random_state=random_state, coef=True, **kwargs
            )
        else:
            if kwargs["n_features"] < 5:
                kwargs["n_features"] = 5
            X, y = datasets.make_friedman1(random_state=random_state, **kwargs)
            coef = None
        Xs[key] = X
        ys[key] = y
        coefs[key] = coef

    return Xs, ys, coefs


def plot_3d_projection(iterator, Xs, ys):
    """Plots the 3D projection of the datasets on a facet grid.

    Parameters
    ----------
    iterator : str
        The argument (as a list) to iterate over. Must be one of `n_samples`, `n_features`, `noise`.
    Xs : dictionary of ndarrays (n_samples, n_features)
        The input samples for each value of the iterator.
    ys : dictionary of ndarrays (n_samples,)
        The output samples for each value of the iterator.
    """

    keys = Xs.keys()
    nrows = int(np.ceil(len(keys) / 3))

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(15, 5 * nrows),
        sharex=True,
        sharey=True,
        subplot_kw=dict(projection="3d"),
    )

    for ax, key in zip(axs.flatten(), keys):

        y_plot = ys[key]
        if iterator == "n_features":
            pca = PCA(n_components=3)
            X_plot = pca.fit_transform(Xs[key])
            label = "PC"
        else:
            X_plot = Xs[key]
            label = "X"

        ax.scatter(
            X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, s=10, cmap="RdYlBu_r"
        )
        ax.set_title(f"{iterator}: {key}")
        ax.view_init(elev=20, vertical_axis="z")
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")

    return None
