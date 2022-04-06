import operator
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.example_libraries.optimizers as optimizers
import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as stats
from tqdm.auto import tqdm

from gaul.types import PRNGKey, Pytree
from gaul.utils.tree_utils import tree_zeros_like


@jax.jit
def diag_gaussian_sample(rng: PRNGKey, mean: Pytree, log_std: Pytree) -> Pytree:
    return jax.tree_util.tree_map(
        lambda m, ls: random.normal(rng, m.shape) * jnp.exp(ls) + m, mean, log_std
    )


@jax.jit
def diag_gaussian_logpdf(x: jnp.ndarray, mean: Pytree, log_std: Pytree) -> float:
    logpdfs = jax.tree_util.tree_map(
        lambda x, m, ls: jax.vmap(stats.norm.logpdf)(x, m, jnp.exp(ls)),
        x,
        mean,
        log_std,
    )
    return jax.tree_util.tree_reduce(operator.add, logpdfs, 0.0)


@partial(jax.jit, static_argnums=(0,))
def elbo(ln_prob, rng: PRNGKey, mean: Pytree, log_std: Pytree) -> float:
    sample = diag_gaussian_sample(rng, mean, log_std)
    return ln_prob(sample) - diag_gaussian_logpdf(sample, mean, log_std)


def batch_elbo(ln_prob: Callable, rng: PRNGKey, vi_params: Any, nsamples: int) -> float:
    # Average over a batch of random samples.
    rngs = random.split(rng, nsamples)
    vectorized_elbo = jax.vmap(partial(elbo, ln_prob), in_axes=(0, None, None))
    return jnp.mean(vectorized_elbo(rngs, *vi_params))


def sample(
    ln_posterior: Callable,
    params: Pytree,
    nsteps: int = 10000,
    nsamples: int = 2000,
    rng: PRNGKey = random.PRNGKey(0),
    opt: Optional[optimizers.Optimizer] = None,
    lr: Optional[float] = None,
    *args,
    **kwargs,
) -> Pytree:
    """
    Run mean-field variational inference to sample from the posterior
    distribution.
    """

    if not opt:
        if lr:
            opt = optimizers.adam(lr)
        else:
            opt = optimizers.adam(1e-2)

    ln_prob = jax.jit(lambda x: ln_posterior(x, *args, **kwargs))

    means = tree_zeros_like(params)
    log_stds = tree_zeros_like(params)

    init_vi_params = (means, log_stds)
    opt_init, opt_update, get_params = opt
    opt_state = opt_init(init_vi_params)

    @jax.jit
    def objective(vi_params, t):
        rng = random.PRNGKey(t)
        return -batch_elbo(ln_prob, rng, vi_params, nsamples)

    @jax.jit
    def update(opt_state, i):
        vi_params = get_params(opt_state)
        gradient = jax.grad(objective)(vi_params, i)
        return opt_update(i, gradient, opt_state)

    for i in (pbar := tqdm(range(nsteps))):
        opt_state = update(opt_state, i)
        vi_params = get_params(opt_state)
        if i % (nsteps // 20) == 0:
            pbar.set_description(f"ELBO: {-objective(vi_params, i):.2e}")

    rngs = random.split(rng, nsamples)
    samples = jax.vmap(diag_gaussian_sample, in_axes=(0, None, None))(rngs, *vi_params)
    samples = jax.tree_util.tree_map(lambda x: x.reshape(-1, nsamples), samples)

    return samples
