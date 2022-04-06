from typing import Callable, Iterable

import jax
import jax.numpy as jnp

from gaul.types import PRNGKey, Pytree


def tree_zeros_like(tree: Pytree) -> Pytree:
    """
    Return a new tree with the same structure as t, but with all values set to
    0.
    """
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)


def tree_ones_like(tree: Pytree) -> Pytree:
    """
    Return a new tree with the same structure as t, but with all values set to
    1.
    """
    return jax.tree_util.tree_map(lambda x: jnp.ones_like(x), tree)


def tree_split_keys_like(key: PRNGKey, tree: Pytree) -> Pytree:
    """
    Split the key into multiple keys, one for each leaf of the tree.
    """
    treedef = jax.tree_util.tree_structure(tree)
    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(key: PRNGKey, tree: Pytree, mean=0.0, std=1.0) -> Pytree:
    """
    Return a new tree with the same structure as t, but with all values set to
    random normal variates.
    """
    keys_tree = tree_split_keys_like(key, tree)
    return jax.tree_util.tree_multimap(
        lambda l, k: std * (jax.random.normal(k, l.shape, l.dtype) + mean),
        tree,
        keys_tree,
    )


def tree_stack(trees: Iterable[Pytree], axis: int = 0) -> Pytree:
    """
    Stack a list of trees along a given axis.
    """
    return jax.tree_util.tree_multimap(lambda *x: jnp.stack(x, axis=axis), *trees)


def make_tree_hessian(hess_fn: Callable) -> Callable:
    """
    Makes a function that computes a block diagonal Hessian of a function on a
    tree. The blocks are organized into the same structure as the tree.

    Args:
        hess_fn: A function that takes a tree and returns a tree of Hessians.
                 This can be made with, for example, jax.hessian(fn).
    """

    def tree_hessian(params):
        treedef = jax.tree_util.tree_structure(params)
        hessian = hess_fn(params)
        hessian_leaves = [j[i] for i, j in hessian.items()]
        hessian_tree = jax.tree_util.tree_unflatten(treedef, hessian_leaves)
        return hessian_tree

    return tree_hessian


def dense_hessian(
    ln_posterior: Callable, params: Pytree, *args, **kwargs
) -> jnp.ndarray:
    def flatten(v):
        def f(v):
            leaves, _ = jax.tree_util.tree_flatten(v)
            return jnp.concatenate([x.ravel() for x in leaves])

        out, pullback = jax.vjp(f, v)
        return out, lambda x: pullback(x)[0]

    flat_params, unflatten = flatten(params)
    return jax.hessian(lambda p: ln_posterior(unflatten(p), *args, **kwargs))(
        flat_params
    )
