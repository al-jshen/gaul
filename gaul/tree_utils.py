from typing import Iterable

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
