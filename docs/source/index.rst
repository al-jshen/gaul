Gaul
================================

Quickstart
-----------

To sample 100 chains from a 10D Gaussian:

.. code-block:: python

    import jax.numpy as jnp
    import jax.scipy.stats as stats
    from gaul.modelling import hmc

    def ln_posterior(params):
        return stats.norm.logpdf(params).sum()

    params = jnp.zeros(10)
    samples = hmc.sample(ln_posterior, params, n_chains=100, step_size=0.2)


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

.. toctree::
   :maxdepth: 3
   :caption: Notes

   user/gaul
   changelog

License
-------

Copyright 2022, Jeff Shen.

The source code is free software dual licensed under MIT and Apache 2.0. For more details, see the ``LICENSE-MIT`` and/or ``LICENSE-APACHE`` files.
