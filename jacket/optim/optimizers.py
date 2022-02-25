from typing import Callable

import jax

from jacket.utils import Pytree


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
