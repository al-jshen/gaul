import jax


def sgd(fn, params, lr, niter, *args, **kwargs):
    @jax.jit
    def update(params, *args, **kwargs):
        grads = jax.grad(fn)(params, *args, **kwargs)
        return jax.tree_util.tree_multimap(lambda p, g: p - lr * g, params, grads)

    for _ in range(niter):
        params = update(params, *args, **kwargs)

    return params
