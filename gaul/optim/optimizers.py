from functools import partial
from typing import Callable

import jax

from gaul.types import Pytree
from gaul.utils.tree_utils import tree_zeros_like


def sgd(
    fn: Callable, params: Pytree, lr: float = 1e-3, niter: int = 500, *args, **kwargs
) -> Pytree:
    @jax.jit
    def update(params, *args, **kwargs):
        grads = jax.grad(fn)(params, *args, **kwargs)
        return jax.tree_util.tree_multimap(lambda p, g: p - lr * g, params, grads)

    for _ in range(niter):
        params = update(params, *args, **kwargs)

    return params


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
    @partial(jax.jit, static_argnames=["nesterov"])
    def update(params, velocity, nesterov=False, *args, **kwargs):
        if nesterov:
            alt_params = jax.tree_util.tree_multimap(
                lambda p, v: p - momentum * v, params, velocity
            )
            grads = jax.grad(fn)(alt_params, *args, **kwargs)
        else:
            grads = jax.grad(fn)(params, *args, **kwargs)

        velocity = jax.tree_util.tree_multimap(
            lambda v, g: momentum * v + lr * g, velocity, grads
        )
        return jax.tree_util.tree_multimap(lambda p, v: p - v, params, velocity)

    velocity = tree_zeros_like(params)
    for _ in range(niter):
        params = update(params, velocity, nesterov=nesterov, *args, **kwargs)

    return params
