import jax
import jax.numpy as jnp
import jax.nn as nn
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
        self.key = key or jax.random.PRNGKey(0)

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

    def __call__(self, x, training: bool, mask=None):
        attn = self.mha(x, x, x, mask)
        if training:
            attn = self.drop1(attn)
        attn = self.layernorm1(attn + x)

        ffn = self.ffn(attn)
        if training:
            ffn = self.drop2(ffn)
        return self.layernorm2(ffn + attn)
