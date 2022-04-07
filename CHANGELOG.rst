Changelog
============

0.4.2 (2022-04-07)
++++++++++++++++++
- Fix progress bar compatibility in ADVI code
- Add more documentation and infrastructure

0.4.1 (2022-04-06)
++++++++++++++++++
- Lower minimum Python version to 3.7.1
- Make sampling interfaces more consistent
- Add more documentation

0.4.0 (2022-04-05)
++++++++++++++++++
- Add mean-field variational inference

0.3.0 (2022-04-05)
++++++++++++++++++
- Add quadratic/Laplace approximation implementation

0.2.2 (2022-04-05)
++++++++++++++++++
- Restructure repository
- Nuke ``nn`` and ``optim`` modules

0.2.1 (2022-02-27)
++++++++++++++++++
- Add mass matrix to HMC (just identity for now)

0.2.0 (2022-02-26)
++++++++++++++++++
- Rewrite HMC sampler

  - Use ``lax.scan`` (a lot faster) with progress bars
  - Allow ``ln_posterior`` function to take additional arguments

- Rewrite optimizers to use ``lax.fori_loop``
- Add more documentation
- Add more tests

0.1.0 (2022-02-25)
++++++++++++++++++
- Initial release
