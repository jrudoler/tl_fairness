"""Optional JAX/GPU backend for the TMLE coverage simulations.

This module accelerates the *many-replicate* coverage studies where the
nuisances are simple logistic regressions. The entire per-replicate pipeline --
IRLS fit of the outcome and propensity models, clever covariate, Newton
fluctuation, EIF variance and Wald CI -- is written with fixed array shapes so
it can be ``jax.vmap``-ed across replicates and ``jax.jit``-compiled into a
single (GPU-capable) kernel.

Where JAX helps: the coverage simulations (logistic nuisances, thousands of
replicates). Where it does NOT: the real-data SuperLearner path (HGB/RF/SVC/MLP),
which stays in sklearn/numpy -- this module is not used there.

JAX is an optional dependency. Importing this module without JAX installed sets
``HAS_JAX = False`` and the public builders raise a clear error; the numpy
reference in :mod:`tlfair.tmle` is always the canonical fallback.

Reproducibility: the data-generating process stays in numpy
(``np.random.default_rng`` + ``spawn``) so draws are bit-for-bit reproducible;
this module only consumes the resulting fixed-shape ``(R, n, d)`` tensors and
performs deterministic numeric computation. x64 is enabled to match the numpy
CIs.
"""

from functools import partial

try:  # pragma: no cover - exercised indirectly via HAS_JAX
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False


def _require_jax():
    if not HAS_JAX:
        raise ImportError(
            "the JAX backend requires JAX; install with `uv pip install "
            "'tl-fairness[jax]'` (or `[jax-cuda]` for GPU). The numpy reference "
            "in tlfair.tmle is always available as a fallback."
        )


if HAS_JAX:

    def expit(z):
        return jax.scipy.special.expit(z)

    def logit(p, eps=1e-6):
        p = jnp.clip(p, eps, 1.0 - eps)
        return jnp.log(p) - jnp.log1p(-p)

    def _add_intercept(X):
        return jnp.concatenate([jnp.ones((X.shape[0], 1), X.dtype), X], axis=1)

    def logreg_irls(X, y, *, n_steps=12, l2=1e-6):
        """Logistic regression via fixed-iteration Newton/IRLS (jit/vmap-safe).

        ``X`` should NOT include an intercept column; one is prepended here.
        Fixed ``n_steps`` (no data-dependent loop) keeps the function traceable.
        """
        Xi = _add_intercept(X)
        d = Xi.shape[1]
        beta = jnp.zeros(d, Xi.dtype)
        ridge = l2 * jnp.eye(d, dtype=Xi.dtype)
        y = y.astype(Xi.dtype)

        def step(beta, _):
            p = expit(Xi @ beta)
            W = jnp.clip(p * (1.0 - p), 1e-8, None)
            grad = Xi.T @ (p - y) + l2 * beta
            H = (Xi * W[:, None]).T @ Xi + ridge
            beta = beta - jnp.linalg.solve(H, grad)
            return beta, None

        beta, _ = jax.lax.scan(step, beta, None, length=n_steps)
        return beta

    def _predict_proba1(beta, X):
        return expit(_add_intercept(X) @ beta)

    def clever_covariate_parity(pi1, p_g1):
        return pi1 / p_g1 - (1.0 - pi1) / (1.0 - p_g1)

    def fit_fluctuation(y, d_hat, H, *, n_steps=12):
        """Scalar logistic-fluctuation parameter via fixed-iteration Newton."""
        ld = logit(d_hat)
        y = y.astype(d_hat.dtype)

        def step(eps, _):
            d_star = expit(ld + eps * H)
            score = jnp.sum(H * (y - d_star))
            info = jnp.sum(H ** 2 * d_star * (1.0 - d_star)) + 1e-12
            return eps + score / info, None

        eps, _ = jax.lax.scan(step, 0.0, None, length=n_steps)
        return eps

    def tmle_replicate_parity(X, g, y, n_tr, *, n_steps=12):
        """One replicate of single-split TMLE for probabilistic parity.

        ``n_tr`` is a STATIC python int (closed over / passed via partial) so the
        slice shapes are concrete under jit/vmap. Returns (est, se, lo, hi).
        """
        Xtr, Xte = X[:n_tr], X[n_tr:]
        ytr, yte = y[:n_tr], y[n_tr:]
        gte = g[n_tr:]
        gtr = g[:n_tr]

        beta_y = logreg_irls(Xtr, ytr, n_steps=n_steps)
        beta_g = logreg_irls(Xtr, gtr, n_steps=n_steps)
        d_hat = _predict_proba1(beta_y, Xte)
        pi1 = _predict_proba1(beta_g, Xte)

        p_g1 = jnp.mean(gte)
        H = clever_covariate_parity(pi1, p_g1)
        eps = fit_fluctuation(yte, d_hat, H, n_steps=n_steps)
        d_star = expit(logit(d_hat) + eps * H)

        plug1 = (gte == 1) * d_star / p_g1
        plug0 = (gte == 0) * d_star / (1.0 - p_g1)
        est = jnp.mean(plug1) - jnp.mean(plug0)
        eif = H * (yte - d_star) + plug1 - plug0 - est
        se = jnp.sqrt(jnp.var(eif) / eif.shape[0])
        return est, se, est - 1.96 * se, est + 1.96 * se


def build_batched_parity(n_tr, *, n_steps=12):
    """Return a jit-compiled function mapping replicate batches to (est,se,lo,hi).

    Inputs to the returned function: ``X:(R, n, d)``, ``g:(R, n)``, ``y:(R, n)``
    with a fixed per-replicate sample size ``n`` and train size ``n_tr``.
    """
    _require_jax()
    fn = partial(tmle_replicate_parity, n_tr=n_tr, n_steps=n_steps)
    return jax.jit(jax.vmap(fn))


def coverage_parity_jax(X, g, y, n_tr, truth, *, n_steps=12):
    """Convenience: run the batched kernel and return (coverage, bias, mean_var).

    ``X,g,y`` are numpy or jax arrays of shape (R,n,d)/(R,n); ``truth`` is the
    Monte-Carlo target. Coverage/bias are computed on host.
    """
    _require_jax()
    est, se, lo, hi = build_batched_parity(int(n_tr), n_steps=n_steps)(
        jnp.asarray(X), jnp.asarray(g), jnp.asarray(y)
    )
    est = jnp.asarray(est)
    coverage = float(jnp.mean((lo <= truth) & (truth <= hi)))
    bias = float(jnp.mean(est - truth))
    var = float(jnp.mean(se ** 2) * X.shape[1])  # E[se^2]*n ~ EIF variance scale
    return coverage, bias, var
