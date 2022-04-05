from typing import Callable

import jax
import jax.example_libraries.optimizers as optimizers
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.flatten_util import ravel_pytree

from gaul.types import PRNGKey, Pytree
from gaul.utils.pbar import progress_bar_scan
from gaul.utils.tree_utils import dense_hessian


def sample(
    ln_posterior: Callable,
    params: Pytree,
    opt: optimizers.Optimizer = optimizers.adam(1e-3),
    nsteps: int = 5000,
    nsamples: int = 1000,
    rng: PRNGKey = random.PRNGKey(0),
    *args,
    **kwargs,
) -> jnp.ndarray:
    opt_init, opt_update, get_params = opt
    opt_state = opt_init(params)

    neg_ln_posterior = lambda p: -ln_posterior(p, *args, **kwargs)

    @jax.jit
    def update(opt_state, i):
        params = get_params(opt_state)
        gradient = jax.grad(neg_ln_posterior)(params)
        return opt_update(i, gradient, opt_state), None

    pbar = progress_bar_scan(nsteps, f"Running {nsteps} optimization steps")

    opt_state, _ = lax.scan(pbar(update), opt_state, jnp.arange(nsteps))
    params = get_params(opt_state)

    hessian = dense_hessian(neg_ln_posterior, params)
    hessian_inv = jnp.linalg.inv(hessian)

    params_flat, params_ravel = ravel_pytree(params)

    samples = random.multivariate_normal(
        rng, params_flat, hessian_inv, shape=(nsamples,)
    )
    samples = jax.vmap(params_ravel)(samples)
    samples = jax.tree_util.tree_map(lambda x: x.reshape(-1, nsamples), samples)

    return samples
