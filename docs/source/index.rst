Gaul
======

Quickstart
-----------

To sample from a 10 dimensional Gaussian with HMC, ADVI, or Laplace/quadratic approximation:

.. code-block:: python

    import jax.numpy as jnp
    import jax.scipy.stats as stats

    from gaul import hmc, advi, quap

    def ln_posterior(params):
        return stats.norm.logpdf(params).sum()

    params = jnp.zeros(10)

    samples_hmc = hmc.sample(
      ln_posterior,
      params,
      step_size=0.2
      n_chains=1000,
      n_warmup=1000,
      n_samples=100
    )

    samples_advi = advi.sample(
      ln_posterior,
      params,
      lr=0.2,
      n_steps=1000,
    )

    samples_quap = quap.sample(ln_posterior, params)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install

.. toctree::
   :maxdepth: 2
   :caption: Examples and Tutorials

   examples/linreg.ipynb
   examples/8schools.ipynb
   examples/autoregressive.ipynb
   examples/gp.ipynb

.. toctree::
   :maxdepth: 3
   :caption: Notes

   user/gaul
   changelog

License
-------

Copyright 2022, Jeff Shen.

The source code is free software dual licensed under MIT and Apache 2.0. For more details, see the ``LICENSE-MIT`` and/or ``LICENSE-APACHE`` files.
