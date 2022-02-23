import jax
import jax.numpy as jnp
import jax.nn as nn


@jax.jit
def scaled_dot_product_attention(Q, K, V):
    assert Q.shape == K.shape
    d_k = jnp.shape(K)[-1]
    KQ_scaled = Q @ K.T / jnp.sqrt(d_k)
    return nn.softmax(KQ_scaled) @ V


@jax.jit
def head(Q, K, V, Wq, Wk, Wv):
    assert Wq.shape == Wk.shape
    return scaled_dot_product_attention(Q @ Wq, K @ Wk, V @ Wv)


batch_vmap_head = jax.jit(jax.vmap(head, in_axes=[None, None, None, 0, 0, 0]))
batch_pmap_head = jax.pmap(head, in_axes=[None, None, None, 0, 0, 0])


def _multihead_attention(Q, K, V, Wq_arr, Wk_arr, Wv_arr, Wo, pmap=False):
    h = Wq_arr.shape[0]
    assert Wk_arr.shape[0] == h
    assert Wv_arr.shape[0] == h

    if pmap:
        fn = batch_pmap_head
    else:
        fn = batch_vmap_head

    return jnp.reshape(fn(Q, K, V, Wq_arr, Wk_arr, Wv_arr), (*Q.shape[:-1], -1)) @ Wo


multihead_attention = jax.jit(_multihead_attention, static_argnames=["pmap"])


@jax.jit
def ffn(x, W1, W2, b1, b2):
    return nn.relu(x @ W1 + b1) @ W2 + b2


@jax.jit
def layernorm(x, gamma, beta, eps=1e-3):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta
