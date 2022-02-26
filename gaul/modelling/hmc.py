from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from tqdm import tqdm

from gaul.types import Pytree
from gaul.utils.tree_utils import tree_random_normal_like, tree_stack


@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, 0, None))
def factor(
    params: Pytree,
    momentum: Pytree,
    ln_posterior: Callable[..., float],
) -> float:
    m = jax.tree_util.tree_reduce(lambda acc, x: acc + x.T @ x, momentum, 0.0)
    return ln_posterior(params) - 0.5 * m


@jax.jit
@jax.vmap
def select(mask, new, old):
    return jax.tree_util.tree_multimap(
        lambda _new, _old: jnp.where(mask, _new, _old), new, old
    )


@partial(jax.jit, static_argnums=(2,))
def accept_reject(
    state_old: Tuple[Pytree, Pytree],
    state_new: Tuple[Pytree, Pytree],
    ln_posterior: Callable[..., float],
    key,
) -> Tuple[Pytree, Pytree]:
    params_old, momentum_old = state_old
    params_new, momentum_new = state_new

    factor_old = factor(params_old, momentum_old, ln_posterior)
    factor_new = factor(params_new, momentum_new, ln_posterior)
    log_accept = factor_new - factor_old
    log_uniform = jnp.log(jax.vmap(jax.random.uniform)(key))
    accept_mask = log_uniform < log_accept

    flipped_momentum_new = jax.tree_util.tree_map(lambda x: -x, momentum_new)

    params = select(accept_mask, params_new, params_old)
    momentum = select(accept_mask, flipped_momentum_new, momentum_old)

    return params, momentum


def leapfrog_step(
    params: Pytree,
    momentum: Pytree,
    step_size: float,
    grad_fn: Callable,
) -> Tuple[Pytree, Pytree]:
    momentum = jax.tree_util.tree_multimap(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params)
    )
    params = jax.tree_util.tree_multimap(
        lambda p, m: p + m * step_size, params, momentum
    )
    momentum = jax.tree_util.tree_multimap(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params)
    )
    return params, momentum


@partial(jax.jit, static_argnums=(4,))
@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def leapfrog(
    params: Pytree,
    momentum: Pytree,
    n_steps: int,
    step_size: float,
    grad_fn: Callable,
) -> Tuple[Pytree, Pytree]:
    return lax.fori_loop(
        0,
        n_steps,
        lambda _, pm: leapfrog_step(pm[0], pm[1], step_size, grad_fn),
        (params, momentum),
    )


@jax.jit
def generate_momentum(key, tree):
    key, subkey = jax.random.split(key)
    momentum = tree_random_normal_like(subkey, tree)
    return key, momentum


def sample(
    ln_posterior: Callable[..., float],
    init_params: Pytree,
    n_chains: int = 4,
    leapfrog_steps: int = 10,
    step_size: float = 1e-3,
    n_samples: int = 1000,
    n_warmup: int = 1000,
    key=None,
) -> Tuple[Pytree, Pytree]:
    if key is None:
        key = jax.random.PRNGKey(0)

    samples = []
    samples_momentum = []

    ln_posterior_grad = jax.jit(jax.grad(ln_posterior))

    params = tree_stack([init_params.copy() for _ in range(n_chains)])

    for i in tqdm(range(n_samples + n_warmup)):

        key, momentum = generate_momentum(key, params)

        params_new, momentum_new = leapfrog(
            params,
            momentum,
            leapfrog_steps,
            step_size,
            ln_posterior_grad,
        )

        keys = jax.random.split(key, n_chains + 1)
        key = keys[0]
        params, momentum = accept_reject(
            (params, momentum),
            (params_new, momentum_new),
            ln_posterior,
            keys[1:],
        )

        if i >= n_warmup:
            samples.append(params)
            samples_momentum.append(momentum)

    return tree_stack(samples, -1), tree_stack(samples_momentum, -1)
