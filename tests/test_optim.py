import jax.numpy as jnp

import gaul.optim.optimizers as optimizers


class TestSGD:
    def test_1d(self):
        loss = lambda x: jnp.sum(x**2)
        x = jnp.array([1.0])
        estimate = optimizers.sgd(loss, x, 0.01, 1000)
        solution = 0.0
        assert jnp.isclose(estimate, solution)

    def test_rosenbrock(self):
        a = 1.0
        b = 10.0
        loss = lambda x: (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
        x = jnp.zeros(2)
        estimate = optimizers.momentum(loss, x, 0.01, 5000, 0.9, True)
        solution = jnp.array([a, a**2])
        assert jnp.isclose(estimate, solution).all()
