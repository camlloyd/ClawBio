"""Null single-cell data with real between-donor variability and no condition effect.

The design follows the null simulations in Zimmerman, Espeland & Langefeld,
Nat Commun 12:738 (2021) and Squair et al., Nat Commun 12:5692 (2021): cells are
nested in donors, every gene's mean varies from donor to donor, and donors (not
cells) are split at random into "control" and "treated". Nothing differs between
conditions, so every gene called differentially expressed is a false discovery.

A test that treats cells as independent replicates sees thousands of "samples"
where there are only a handful of donors, and calls donor-to-donor differences
condition effects. A test at the donor level (pseudobulk) should not.
"""

from __future__ import annotations

import numpy as np


def simulate_null_replicates(
    *,
    donors_per_condition: int = 3,
    cells_per_donor: int = 200,
    n_genes: int = 400,
    donor_log_sd: float = 0.4,
    dispersion: float = 0.5,
    seed: int = 0,
):
    """Return an AnnData of raw counts with obs ``donor``, ``condition`` and ``cell_type``.

    donor_log_sd is the standard deviation of each donor's log-scale shift in a
    gene's mean (between-replicate heterogeneity); dispersion is the negative
    binomial overdispersion of cells around their donor's mean.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_donors = 2 * donors_per_condition
    base_mean = rng.gamma(shape=0.6, scale=3.0, size=n_genes) + 0.05
    donor_shift = rng.normal(0.0, donor_log_sd, size=(n_donors, n_genes))

    conditions = np.array(["control"] * donors_per_condition + ["treated"] * donors_per_condition)
    rng.shuffle(conditions)

    counts, donors, cond = [], [], []
    size = 1.0 / dispersion
    for d in range(n_donors):
        library = rng.lognormal(mean=0.0, sigma=0.3, size=cells_per_donor)
        mu = library[:, None] * base_mean[None, :] * np.exp(donor_shift[d])[None, :]
        counts.append(rng.negative_binomial(size, size / (size + mu)))
        donors += [f"donor{d + 1}"] * cells_per_donor
        cond += [conditions[d]] * cells_per_donor

    obs = pd.DataFrame(
        {"donor": donors, "condition": cond, "cell_type": "T_cell"},
        index=[f"cell{i}" for i in range(len(donors))],
    )
    var = pd.DataFrame(index=[f"gene{g}" for g in range(n_genes)])
    adata = ad.AnnData(X=np.vstack(counts).astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata
