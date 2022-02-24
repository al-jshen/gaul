import jax
import jax.numpy as jnp
from tqdm import tqdm


def factor(params, momentum, ln_posterior, inv_mass):
    return ln_posterior(params) - 0.5 * momentum.T @ inv_mass @ momentum


def accept_reject(state_old, state_new, ln_posterior, inv_mass, key):
    params_old, momentum_old = state_old
    params_new, momentum_new = state_new

    factor_old = factor(params_old, momentum_old, ln_posterior, inv_mass)
    factor_new = factor(params_new, momentum_new, ln_posterior, inv_mass)
    p_accept = jnp.clip(jnp.exp(factor_new - factor_old), a_max=1)

    if jax.random.uniform(key) < p_accept:
        return params_new, -momentum_new, p_accept, 1
    else:
        return params_old, momentum_old, p_accept, 0


def leapfrog(params_init, momentum_init, n_steps, stepsize, grad_fn, inv_mass):
    params, momentum = params_init, momentum_init
    for _ in range(n_steps):
        momentum = momentum + grad_fn(params) * 0.5 * stepsize
        params = params + inv_mass @ momentum * stepsize
        momentum = momentum + grad_fn(params) * 0.5 * stepsize
    return params, momentum


def opt(params_init, n_steps, lr, grad_fn, hess_fn):
    params = params_init
    for _ in tqdm(range(n_steps)):
        grad = grad_fn(params)
        hess = hess_fn(params)
        inv_hess = jnp.linalg.inv(hess)
        params -= lr * inv_hess @ grad
    return params


def sample(
    ln_posterior,
    n_samples,
    n_steps,
    stepsize,
    n_steps_opt,
    stepsize_opt,
    n_params,
    key,
):
    samples = []
    samples_momentum = []
    samples_paccept = []

    ln_posterior_grad = jax.jit(jax.grad(ln_posterior))
    ln_posterior_hessian = jax.jit(jax.hessian(ln_posterior))

    params = jnp.zeros(n_params)
    params = opt(
        params,
        n_steps_opt,
        stepsize_opt,
        ln_posterior_grad,
        ln_posterior_hessian,
    )
    max_params = params

    mass = jnp.zeros((n_params, n_params))

    for _ in tqdm(range(n_samples)):

        mass = ln_posterior_hessian(params) * 1 / n_steps + mass * (1 - 1 / n_steps)
        mass = mass / mass.sum()
        inv_mass = jnp.linalg.inv(mass)

        key, subkey = jax.random.split(key)
        momentum = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(n_params), cov=mass
        )

        params_new, momentum_new = leapfrog(
            params, momentum, n_steps, stepsize, ln_posterior_grad, inv_mass
        )

        key, subkey = jax.random.split(key)
        params, momentum, p_accept, _ = accept_reject(
            (params, momentum),
            (params_new, momentum_new),
            ln_posterior,
            inv_mass,
            subkey,
        )

        samples.append(params)
        samples_momentum.append(momentum)
        samples_paccept.append(p_accept)

    return (
        jnp.array(samples),
        jnp.array(samples_momentum),
        jnp.array(samples_paccept),
        max_params,
    )
