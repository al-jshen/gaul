from typing import Any, Callable, Iterable
import jax
import jax.numpy as jnp
from functools import partial


class Sequential:
    def __init__(self, layers: Iterable[Callable[..., Any]]):
        self.layers = layers

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, inputs, *args, **kwargs):
        out = inputs
        for layer in self.layers:
            out = layer(out, *args, **kwargs)
        return out


class Linear:
    def __init__(self, input_size: int, output_size: int, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.weights = jax.random.normal(key, (input_size, output_size))
        self.bias = jnp.zeros(output_size)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        return x @ self.weights + self.bias


class Dense(Linear):
    def __init__(self, input_size: int, output_size: int, activation=None, key=None):
        super().__init__(input_size, output_size, key)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation or (lambda x: x)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        return self.activation(super().__call__(x))


class Dropout:
    def __init__(self, p: float, key=None):
        self.p = jnp.clip(p, 0.0, 1.0)
        self.key = jax.random.PRNGKey(0) if key is None else key

    @partial(jax.jit, static_argnums=(0, 2))
    def __call__(self, x, training=False):
        if training:
            if self.p == 0.0:
                return x
            self.key, subkey = jax.random.split(self.key)
            keep_rate = 1.0 - self.p
            keep = jax.random.bernoulli(subkey, keep_rate, shape=x.shape)
            return keep * x / keep_rate
        else:
            return x


class LayerNorm:
    def __init__(self, features, eps=1e-3):
        super().__init__()
        self.gamma = jnp.ones(features)
        self.beta = jnp.zeros(features)
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
