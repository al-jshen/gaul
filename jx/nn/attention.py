from typing import Optional
from functools import partial
import jax
import jax.numpy as jnp
import jax.nn as nn
from jx.nn.utils import LayerNorm, Linear, Sequential, Dropout, Dense


@jax.jit
def scaled_dot_product_attention(Q, K, V, mask=None):
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    d_k = K.shape[-1]
    assert Q.shape[-1] == d_k

    scores = Q @ K.T
    scores_scaled = scores / jnp.sqrt(d_k)

    if mask is not None:
        scores_scaled += mask * -1e10

    weights = nn.softmax(scores_scaled)
    return weights @ V


batch_vmap_scaled_dot_product_attention = jax.jit(
    jax.vmap(scaled_dot_product_attention, in_axes=[0, 0, 0, None])
)


class MultiheadAttention:
    def __init__(
        self,
        num_heads: int,
        model_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        key=None,
    ) -> None:
        assert model_size % num_heads == 0
        self.num_heads = num_heads
        self.model_size = model_size
        self.key_size = key_size or model_size // num_heads
        self.value_size = value_size or self.key_size
        if key is None:
            self.key = jax.random.PRNGKey(0)
        else:
            self.key = key

        assert self.value_size is not None

        k1, k2, k3, k4 = jax.random.split(self.key, 4)
        self.wq = Linear(
            self.num_heads * self.key_size, self.num_heads * self.key_size, key=k1
        )
        self.wk = Linear(
            self.num_heads * self.key_size, self.num_heads * self.key_size, key=k2
        )
        self.wv = Linear(
            self.num_heads * self.value_size, self.num_heads * self.value_size, key=k3
        )
        self.wo = Linear(self.model_size, self.model_size, key=k4)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        assert K.shape[0] == V.shape[0] == batch_size

        q = self.wq(Q)
        k = self.wk(K)
        v = self.wv(V)

        attn = batch_vmap_scaled_dot_product_attention(q, k, v, mask)

        return self.wo(attn)
