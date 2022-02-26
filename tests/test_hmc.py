import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from jacket.modelling import hmc


class TestHMC:
    def test_mvn(self):
        @jax.jit
        def ln_posterior(params):
            return stats.norm.logpdf(params["x"], 0, 1).sum()

        n_dims = 5
        n_chains = 1000
        n_samples = 1000

        params = dict(x=jnp.ones(n_dims))

        samples, momentum = hmc.sample(
            ln_posterior, params, n_chains=n_chains, n_samples=n_samples, step_size=1e-2
        )

        x_samples = samples["x"]

        assert x_samples.shape == (n_chains, n_dims, n_samples)

        def close_to(chain, val, tol=3.0):
            return jnp.abs(jnp.mean(chain) - val) < tol * jnp.std(chain)

        all_close = jax.vmap(close_to, in_axes=(0, None, None))(x_samples, 0.0, 3.0)

        assert all_close.sum() >= n_chains * 0.99
