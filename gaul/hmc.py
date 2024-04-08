from collections import OrderedDict
from functools import partial
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from einops import repeat
from jax import Array, lax

from gaul.types import PRNGKey, Pytree
from gaul.utils.pbar import progress_bar_scan
from gaul.utils.tree_utils import tree_random_normal_like

import math


@partial(jax.jit, static_argnums=(2, 3))
@partial(jax.vmap, in_axes=(0, 0, None, None))
def factor(
    params: Pytree,
    momentum: Pytree,
    ln_posterior: Callable[..., float],
    mass_matrix_fn: Callable,
) -> float:
    tree_momentum_mass = jax.tree_map(
        lambda x, ih: x.T @ ih @ x, momentum, mass_matrix_fn(params)
    )
    m = jax.tree_util.tree_reduce(lambda acc, x: acc + x, tree_momentum_mass, 0.0)
    return ln_posterior(params) - 0.5 * m


@jax.jit
@jax.vmap
def select(mask, new, old):
    return jax.tree_map(lambda _new, _old: jnp.where(mask, _new, _old), new, old)


@partial(jax.jit, static_argnums=(2, 3))
def accept_reject(
    state_old: Tuple[Pytree, Pytree],
    state_new: Tuple[Pytree, Pytree],
    ln_posterior: Callable[..., float],
    mass_matrix_fn: Callable,
    key,
) -> Tuple[Pytree, Pytree, Array]:
    params_old, momentum_old = state_old
    params_new, momentum_new = state_new

    factor_old = factor(params_old, momentum_old, ln_posterior, mass_matrix_fn)
    factor_new = factor(params_new, momentum_new, ln_posterior, mass_matrix_fn)
    log_accept = factor_new - factor_old
    log_uniform = jnp.log(jax.vmap(jax.random.uniform)(key))
    accept_mask = log_uniform < log_accept

    flipped_momentum_new = jax.tree_util.tree_map(lambda x: -x, momentum_new)

    params = select(accept_mask, params_new, params_old)
    momentum = select(accept_mask, flipped_momentum_new, momentum_old)

    return params, momentum, jnp.exp(log_accept)


def leapfrog_step(
    params: Pytree,
    momentum: Pytree,
    step_size: float,
    grad_fn: Callable,
    mass_matrix_fn: Callable,
) -> Tuple[Pytree, Pytree]:
    momentum = jax.tree_map(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params)
    )
    params = jax.tree_map(
        lambda p, m, ih: p + ih @ m * step_size,
        params,
        momentum,
        mass_matrix_fn(params),
    )
    momentum = jax.tree_map(
        lambda m, g: m + g * 0.5 * step_size, momentum, grad_fn(params)
    )
    return params, momentum


@partial(
    jax.jit,
    static_argnums=(
        4,
        5,
    ),
)
@partial(jax.vmap, in_axes=(0, 0, None, 0, None, None))
def leapfrog(
    params: Pytree,
    momentum: Pytree,
    n_steps: int,
    step_size: Pytree,
    grad_fn: Callable,
    mass_matrix_fn: Callable,
) -> Tuple[Pytree, Pytree]:
    return lax.fori_loop(
        0,
        n_steps,
        lambda _, pm: leapfrog_step(pm[0], pm[1], step_size, grad_fn, mass_matrix_fn),
        (params, momentum),
    )


@jax.jit
def generate_momentum(key: PRNGKey, tree: Pytree) -> Tuple[PRNGKey, Pytree]:
    key, subkey = jax.random.split(key)
    momentum = tree_random_normal_like(subkey, tree)
    return key, momentum


def transpose_samples(samples: Pytree, shape: Tuple[int, ...]) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x.transpose(*shape), samples)


def initialize_dual_averaging_params(
    initial_step_size,
    adapt_steps=1000,
    chains=4,
    target_accept=0.80,
    gamma=0.05,
    t0=10.0,
    kappa=0.75,
):
    params = {
        "step_size": initial_step_size,
        "mu": jnp.log(10.0 * initial_step_size),
        "target_accept": target_accept,
        "gamma": gamma,
        "t": t0,
        "kappa": kappa,
        "error_sum": 0.0,
        "log_averaged_step": 0.0,
        "adapt_steps": adapt_steps,
    }
    return jax.tree_util.tree_map(
        lambda x: repeat(jnp.array(x), "... -> c ...", c=chains), params
    )


def _update_dual_averaging_params(params, p_accept):
    error_sum = params["error_sum"] + params["target_accept"] - p_accept
    log_step = params["mu"] - error_sum / (jnp.sqrt(params["t"]) * params["gamma"])
    eta = params["t"] ** (-params["kappa"])
    log_averaged_step = eta * log_step + (1 - eta) * params["log_averaged_step"]
    t = params["t"] + 1
    step_size = jnp.exp(log_step)
    return {
        "step_size": step_size,
        "mu": params["mu"],
        "target_accept": params["target_accept"],
        "gamma": params["gamma"],
        "t": t,
        "kappa": params["kappa"],
        "error_sum": error_sum,
        "log_averaged_step": log_averaged_step,
        "adapt_steps": params["adapt_steps"],
    }


def _finalize_step_size(_params):
    params = _params
    params["step_size"] = jnp.exp(params["log_averaged_step"])
    return params


def update_dual_averaging_params(params, p_accept):
    return jax.lax.cond(
        jnp.all(params["t"] >= params["adapt_steps"]),
        lambda _: _finalize_step_size(params),
        lambda _: _update_dual_averaging_params(params, p_accept),
        operand=None,
    )


# @partial(jax.jit, static_argnums=(0, 2, 5, 6))
def sample(
    ln_posterior: Callable[..., float],
    init_params: Pytree,
    n_chains: int = 4,
    leapfrog_steps: int = 10,
    step_size: float = 1e-3,
    n_samples: int = 1000,
    n_warmup: int = 1000,
    target_accept: float = 0.80,
    key: PRNGKey = jax.random.PRNGKey(0),
    return_momentum: bool = False,
    *args,
    **kwargs,
) -> Union[Pytree, Tuple[Pytree, Pytree]]:
    print("Compiling...")

    dual_averaging_params = initialize_dual_averaging_params(
        step_size, adapt_steps=n_warmup, chains=n_chains, target_accept=target_accept
    )

    if isinstance(init_params, dict):
        init_params = OrderedDict(init_params)

    ln_posterior = jax.jit(partial(ln_posterior, *args, **kwargs))
    ln_posterior_grad = jax.jit(jax.grad(ln_posterior))

    params = jax.tree_util.tree_map(
        lambda x: repeat(x, "... -> c ...", c=n_chains), init_params
    )
    key, momentum_filler = generate_momentum(key, params)

    def mass_matrix_fn(params):
        return jax.tree_util.tree_map(lambda x: jnp.eye(x.size), params)

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def step(
        ln_posterior,
        ln_posterior_grad,
        mass_matrix_fn,
        params,
        _,
        da_params,
        key,
    ):
        key, momentum = generate_momentum(key, params)

        # cant use kwargs here because of weird vmap behaviour
        params_new, momentum_new = leapfrog(
            params,  # params
            momentum,  # momentum
            leapfrog_steps,  # n_steps
            da_params["step_size"],  # step_size
            ln_posterior_grad,  # grad_fn
            mass_matrix_fn,  # mass_matrix_fn
        )

        keys = jax.random.split(key, n_chains + 1)
        key = keys[0]
        params, momentum, p_accept = accept_reject(
            (params, momentum),
            (params_new, momentum_new),
            ln_posterior,
            mass_matrix_fn,
            keys[1:],
        )
        da_params = update_dual_averaging_params(da_params, p_accept)
        return (params, momentum, da_params, key)

    @jax.jit
    def step_carry(carry, _):
        p, m, dap, k = carry
        p, m, dap, k = step(
            ln_posterior, ln_posterior_grad, mass_matrix_fn, p, m, dap, k
        )
        return (p, m, dap, k), (p, m, dap)

    warmup_pbar = progress_bar_scan(n_warmup, f"Running {n_warmup} warmup iterations:")
    sample_pbar = progress_bar_scan(
        n_samples, f"Running {n_samples} sampling iterations:"
    )

    carry = (params, momentum_filler, dual_averaging_params, key)

    print(dual_averaging_params["step_size"])

    carry, params_momentum_dap_warmup = lax.scan(
        warmup_pbar(step_carry), carry, jnp.arange(n_warmup)
    )

    samples_warmup, _, dap_warmup = params_momentum_dap_warmup

    print(
        f"Warmup done. Step sizes adapted. Fixing to final size of {jnp.exp(dap_warmup['log_averaged_step'][-1])}"
    )

    _, params_momentum_dap = lax.scan(
        sample_pbar(step_carry), carry, jnp.arange(n_samples)
    )

    samples, momentum, dual_averaging_params = params_momentum_dap

    if return_momentum:
        return (
            (
                transpose_samples(samples, (1, 2, 0)),
                transpose_samples(momentum, (1, 2, 0)),
            ),
            transpose_samples(samples_warmup, (1, 2, 0)),
            dual_averaging_params,
            dap_warmup,
        )

    return (
        transpose_samples(samples, (1, 2, 0)),
        transpose_samples(samples_warmup, (1, 2, 0)),
        dual_averaging_params,
        dap_warmup,
    )
