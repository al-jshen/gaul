import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from gaul import quap


class TestQuap:
    def test_mvn(self):
        @jax.jit
        def ln_posterior(params):
            return stats.norm.logpdf(params["x"], 0, 1).sum()

        n_dims = 5
        n_samples = 2000

        params = dict(x=jnp.ones(n_dims))

        samples = quap.sample(ln_posterior, params, n_steps=10000, n_samples=n_samples)

        x_samples = samples["x"]

        assert x_samples.shape == (n_dims, n_samples)

        def close_to(chain, val, tol=3.0):
            return jnp.abs(jnp.mean(chain) - val) < tol * jnp.std(chain, axis=0)

        assert close_to(x_samples, 0.0, 3.0).sum() / n_samples >= 0.99
