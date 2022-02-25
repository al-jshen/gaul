from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from jacket.utils import Pytree, tree_random_normal_like, tree_stack


def factor(
    params: Pytree,
    momentum: Pytree,
    ln_posterior: Callable[..., float],
    *args,
    **kwargs,
) -> float:
    m = jax.tree_util.tree_reduce(lambda acc, x: acc + x.T @ x, momentum, 0.0)
    return ln_posterior(params, *args, **kwargs) - 0.5 * m


def accept_reject(
    state_old: Tuple[Pytree, Pytree],
    state_new: Tuple[Pytree, Pytree],
    ln_posterior: Callable[..., float],
    key,
    *args,
    **kwargs,
) -> Tuple[Pytree, Pytree, float, bool]:
    params_old, momentum_old = state_old
    params_new, momentum_new = state_new

    factor_old = factor(params_old, momentum_old, ln_posterior, *args, **kwargs)
    factor_new = factor(params_new, momentum_new, ln_posterior, *args, **kwargs)
    p_accept = jnp.clip(jnp.exp(factor_new - factor_old), a_max=1)

    if jax.random.uniform(key) < p_accept:
        neg_momentum_new = jax.tree_util.tree_map(lambda x: -x, momentum_new)
        return params_new, neg_momentum_new, p_accept, True
    else:
        return params_old, momentum_old, p_accept, False


def leapfrog_step(
    params: Pytree,
    momentum: Pytree,
    step_size: float,
    grad_fn: Callable,
    *args,
    **kwargs,
) -> Tuple[Pytree, Pytree]:
    momentum = jax.tree_util.tree_multimap(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params, *args, **kwargs)
    )
    params = jax.tree_util.tree_multimap(
        lambda p, m: p + m * step_size, params, momentum
    )
    momentum = jax.tree_util.tree_multimap(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params, *args, **kwargs)
    )
    return params, momentum


def leapfrog(
    params_init: Pytree,
    momentum_init: Pytree,
    n_steps: int,
    step_size: float,
    grad_fn: Callable,
    *args,
    **kwargs,
) -> Tuple[Pytree, Pytree]:
    params, momentum = params_init, momentum_init
    for _ in range(n_steps):
        params, momentum = leapfrog_step(
            params, momentum, step_size, grad_fn, *args, **kwargs
        )
    return params, momentum


def sample(
    ln_posterior: Callable[..., float],
    init_params: Pytree,
    leapfrog_steps: int = 10,
    step_size: float = 1e-3,
    n_samples: int = 1000,
    n_warmup: int = 1000,
    key=None,
    *args,
    **kwargs,
) -> Tuple[Pytree, Pytree]:
    if key is None:
        key = jax.random.PRNGKey(0)

    samples = []
    samples_momentum = []

    ln_posterior_grad = jax.jit(jax.grad(ln_posterior))

    params = init_params

    for i in tqdm(range(n_samples + n_warmup)):

        key, subkey = jax.random.split(key)
        momentum = tree_random_normal_like(subkey, params)

        params_new, momentum_new = leapfrog(
            params,
            momentum,
            leapfrog_steps,
            step_size,
            ln_posterior_grad,
            *args,
            **kwargs,
        )

        key, subkey = jax.random.split(key)
        params, momentum, _, _ = accept_reject(
            (params, momentum),
            (params_new, momentum_new),
            ln_posterior,
            subkey,
            *args,
            **kwargs,
        )

        if i >= n_warmup:
            samples.append(params)
            samples_momentum.append(momentum)

    return tree_stack(samples), tree_stack(samples_momentum)
