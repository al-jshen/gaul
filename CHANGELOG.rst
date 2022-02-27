Changelog
============

0.2.0 (2022-02-26)
++++++++++++++++++
* Rewrite HMC sampler
  * Use ``lax.scan`` (a lot faster) with progress bars
  * Allow ``ln_posterior`` function to take additional arguments
* Rewrite optimizers to use ``lax.fori_loop``
* Add more documentation
* Add more tests

0.1.0 (2022-02-25)
++++++++++++++++++
* Initial release
