import jax
from jax._src.flatten_util import ravel_pytree
import jax.numpy as jnp
from tqdm import tqdm


def factor(params, momentum, ln_posterior, inv_mass, *args, **kwargs):
    return (
        ln_posterior(params, *args, **kwargs) - 0.5 * momentum.T @ inv_mass @ momentum
    )


def accept_reject(state_old, state_new, ln_posterior, inv_mass, key, *args, **kwargs):
    params_old, momentum_old = state_old
    params_new, momentum_new = state_new

    factor_old = factor(
        params_old, momentum_old, ln_posterior, inv_mass, *args, **kwargs
    )
    factor_new = factor(
        params_new, momentum_new, ln_posterior, inv_mass, *args, **kwargs
    )
    p_accept = jnp.clip(jnp.exp(factor_new - factor_old), a_max=1)

    if jax.random.uniform(key) < p_accept:
        return params_new, -momentum_new, p_accept, 1
    else:
        return params_old, momentum_old, p_accept, 0


def leapfrog(
    params_init, momentum_init, n_steps, stepsize, grad_fn, inv_mass, *args, **kwargs
):
    params, momentum = params_init, momentum_init
    for _ in range(n_steps):
        momentum = momentum + grad_fn(params, *args, **kwargs) * 0.5 * stepsize
        params = params + inv_mass @ momentum * stepsize
        momentum = momentum + grad_fn(params, *args, **kwargs) * 0.5 * stepsize
    return params, momentum


def sample(
    ln_posterior,
    init_params,
    leapfrog_steps=10,
    step_size=1e-3,
    n_samples=1000,
    n_warmup=1000,
    key=None,
    *args,
    **kwargs,
):
    if key is None:
        key = jax.random.PRNGKey(0)

    samples = []
    samples_momentum = []

    ln_posterior_grad = jax.jit(jax.grad(ln_posterior))

    params, ravel_fn = ravel_pytree(init_params)
    ravel_vmap = jax.vmap(ravel_fn)
    n_params = params.size

    mass = jnp.eye(n_params, n_params)
    inv_mass = jnp.eye(n_params, n_params)

    for i in tqdm(range(n_samples + n_warmup)):

        key, subkey = jax.random.split(key)
        momentum = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(n_params), cov=mass
        )

        params_new, momentum_new = leapfrog(
            params,
            momentum,
            leapfrog_steps,
            step_size,
            ln_posterior_grad,
            inv_mass,
            *args,
            **kwargs,
        )

        key, subkey = jax.random.split(key)
        params, momentum, _, _ = accept_reject(
            (params, momentum),
            (params_new, momentum_new),
            ln_posterior,
            inv_mass,
            subkey,
        )

        if i >= n_warmup:
            samples.append(params)
            samples_momentum.append(momentum)

    return ravel_vmap(jnp.array(samples)), ravel_vmap(jnp.array(samples_momentum))
