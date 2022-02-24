import jax
import jax.numpy as jnp
import jax.nn as nn
from functools import partial
from jx.nn import attention
from jx.nn.utils import Dense, Dropout, LayerNorm, Sequential


class EncoderLayer:
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout_rate=0.5,
        key=None,
    ):
        self.model_size = model_size
        self.hidden_size = feedforward_size
        self.key = jax.random.PRNGKey(0) if key is None else key

        k1, k2, k3, k4, k5 = jax.random.split(self.key, 5)
        self.mha = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k1,
        )
        self.ffn = Sequential(
            [
                Dense(model_size, feedforward_size, activation=nn.relu, key=k2),
                Dense(feedforward_size, model_size, key=k3),
            ]
        )
        self.layernorm1 = LayerNorm(model_size, eps=1e-6)
        self.layernorm2 = LayerNorm(model_size, eps=1e-6)

        self.drop1 = Dropout(dropout_rate, key=k4)
        self.drop2 = Dropout(dropout_rate, key=k5)

    @partial(jax.jit, static_argnums=(0, 2))
    def __call__(self, x, training=False, mask=None):
        attn = self.mha(x, x, x, mask)
        attn = self.drop1(attn, training)
        attn = self.layernorm1(attn + x)

        ffn = self.ffn(attn)
        ffn = self.drop2(ffn, training)
        return self.layernorm2(ffn + attn)


def make_padding_mask(x):
    return (x == 0).astype(jnp.float32)


def make_lookahead_mask(size: int):
    return -jnp.tri(size) + 1.0


def make_positional_encodings(n: int, d: int):
    div_term = jnp.exp(jnp.arange(0, d, 2) * -(jnp.log(10000.0) / d))
    position = jnp.arange(n).reshape(-1, 1)
    pos_div = position * div_term
    pe = jnp.zeros((n, d))
    pe = pe.at[:, 0::2].set(jnp.sin(pos_div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos_div))
    return pe


class DecoderLayer:
    def __init__(
        self, model_size: int, num_heads: int, feedforward_size: int, key=None
    ):
        self.model_size = model_size
        self.hidden_size = feedforward_size
        self.key = jax.random.PRNGKey(0) if key is None else key

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(self.key, 7)
        self.mha1 = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k1,
        )
        self.mha2 = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k2,
        )
        self.ffn = Sequential(
            [
                Dense(model_size, feedforward_size, activation=nn.relu, key=k3),
                Dense(feedforward_size, model_size, key=k4),
            ]
        )
        self.layernorm1 = LayerNorm(model_size, eps=1e-6)
        self.layernorm2 = LayerNorm(model_size, eps=1e-6)
        self.layernorm3 = LayerNorm(model_size, eps=1e-6)

        self.drop1 = Dropout(0.1, key=k5)
        self.drop2 = Dropout(0.1, key=k6)
        self.drop3 = Dropout(0.1, key=k7)

    @partial(jax.jit, static_argnums=(0, 3))
    def __call__(
        self, x, encoder_output, training=False, lookahead_mask=None, padding_mask=None
    ):
        attn1 = self.mha1(x, x, x, lookahead_mask)
        attn1 = self.drop1(attn1, training)
        attn1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(attn1, encoder_output, encoder_output, padding_mask)
        attn2 = self.drop2(attn2, training)
        attn2 = self.layernorm2(attn2 + attn1)

        ffn = self.ffn(attn2)
        ffn = self.drop3(ffn, training)
        return self.layernorm3(ffn + attn2)
