import jax.numpy as jnp


def lower_bound(x, a):
    return jnp.log(x - a)


def lower_bound_inv(y, a):
    return jnp.exp(y) + a


def upper_bound(x, b):
    return jnp.log(b - x)


def upper_bound_inv(y, b):
    return b - jnp.exp(y)


def logit(u):
    return jnp.log(u / (1.0 - u))


def logit_inv(v):
    return 1.0 / (1.0 + jnp.exp(-v))


def lower_and_upper_bound(x, a, b):
    return logit((x - a) / (b - a))


def lower_and_upper_bound_inv(y, a, b):
    return a + (b - a) * logit_inv(y)
