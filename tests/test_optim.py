import jax.numpy as jnp
import pytest

import gaul.optim.optimizers as optimizers


@pytest.mark.parametrize("opt_fn", [optimizers.sgd, optimizers.momentum])
def test_1d(opt_fn):
    loss = lambda x: jnp.sum(x**2)
    x = jnp.array([1.0])
    estimate = opt_fn(loss, x, 0.01, 5000)
    solution = 0.0
    assert jnp.isclose(estimate, solution)


@pytest.mark.parametrize(
    "opt_fn,b", [(optimizers.sgd, 1.0), (optimizers.momentum, 10.0)]
)
def test_rosenbrock(opt_fn, b):
    a = 1.0
    loss = lambda x: (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    x = jnp.zeros(2)
    estimate = opt_fn(loss, x, 0.01, 5000)
    solution = jnp.array([a, a**2])
    assert jnp.linalg.norm(estimate - solution) < 1e-2
