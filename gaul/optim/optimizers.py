from functools import partial
from typing import Callable

import jax
from jax import lax

from gaul.types import Pytree
from gaul.utils.tree_utils import tree_zeros_like


@partial(jax.jit, static_argnums=(0,))
def sgd(
    fn: Callable, params: Pytree, lr: float = 1e-3, niter: int = 500, *args, **kwargs
) -> Pytree:
    """
    Optimizes some parameters `params` under a function `fn` using stochastic
    gradient descent.

    Args:
        fn: The function to optimize.
        params: The parameters to optimize.
        lr: The learning rate.
        niter: The number of iterations to run.
        *args: Additional arguments to pass to `fn`.
        **kwargs: Additional keyword arguments to pass to `fn`.

    Returns:
        The optimized parameters.
    """

    grad_fn = jax.jit(jax.grad(partial(fn, *args, **kwargs)))

    @jax.jit
    def update(params):
        grads = grad_fn(params)
        return jax.tree_util.tree_multimap(lambda p, g: p - lr * g, params, grads)

    return lax.fori_loop(0, niter, lambda _, p: update(p), params)


@partial(jax.jit, static_argnums=(0,))
def momentum_update(fn, params, velocity, lr, momentum, *args, **kwargs):
    """
    Performs a single step of the momentum update.

    Args:
        fn: The function to optimize.
        params: The parameters to optimize.
        velocity: The current velocity.
        lr: The learning rate.
        momentum: The momentum.
        *args: Additional arguments to pass to `fn`.
        **kwargs: Additional keyword arguments to pass to `fn`.

    Returns:
        The updated parameters.
    """
    grads = jax.grad(fn)(params, *args, **kwargs)

    velocity = jax.tree_util.tree_multimap(
        lambda v, g: momentum * v + lr * g, velocity, grads
    )
    return jax.tree_util.tree_multimap(lambda p, v: p - v, params, velocity), velocity


@partial(jax.jit, static_argnums=(0,))
def nesterov_update(fn, params, velocity, lr, momentum, *args, **kwargs):
    """
    Performs a single step of the Nesterov update.

    Args:
        fn: The function to optimize.
        params: The parameters to optimize.
        velocity: The current velocity.
        lr: The learning rate.
        momentum: The momentum.
        *args: Additional arguments to pass to `fn`.
        **kwargs: Additional keyword arguments to pass to `fn`.

    Returns:
        The updated parameters.
    """
    alt_params = jax.tree_util.tree_multimap(
        lambda p, v: p - momentum * v, params, velocity
    )
    grads = jax.grad(fn)(alt_params, *args, **kwargs)

    velocity = jax.tree_util.tree_multimap(
        lambda v, g: momentum * v + lr * g, velocity, grads
    )
    return jax.tree_util.tree_multimap(lambda p, v: p - v, params, velocity), velocity


@partial(jax.jit, static_argnums=(0, 5))
def momentum(
    fn: Callable,
    params: Pytree,
    lr: float = 1e-3,
    niter: int = 500,
    momentum: float = 0.9,
    nesterov: bool = False,
    *args,
    **kwargs,
) -> Pytree:
    """
    Optimizes some parameters `params` under a function `fn` using momentum.

    Args:
        fn: The function to optimize.
        params: The parameters to optimize.
        lr: The learning rate.
        niter: The number of iterations to run.
        momentum: The momentum.
        nesterov: Whether to use Nesterov momentum.
        *args: Additional arguments to pass to `fn`.
        **kwargs: Additional keyword arguments to pass to `fn`.

    Returns:
        The optimized parameters.
    """

    update_fn = nesterov_update if nesterov else momentum_update
    loop_fn = lambda p, m: update_fn(fn, p, m, lr, momentum, *args, **kwargs)

    velocity = tree_zeros_like(params)

    params, velocity = lax.fori_loop(
        0, niter, lambda _, p: loop_fn(p[0], p[1]), (params, velocity)
    )

    return params
