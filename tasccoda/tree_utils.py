"""
Utility functions to help with handling tree objects in tascCODA

:authors: Johannes Ostner
"""
import numpy as np
import toytree as tt
import pandas as pd

from typing import Tuple, List


def get_A(
        tree: tt.tree,
) -> Tuple[np.ndarray, int]:
    """
    Calculate ancestor matrix from a toytree tree

    Parameters
    ----------
    tree
        A toytree tree object

    Returns
    -------
    Ancestor matrix and number of nodes without root node

    A
        Ancestor matrix (numpy array)
    T
        number of nodes in the tree, excluding the root node
    """
    # Builds ancestor matrix

    n_tips = tree.ntips
    n_nodes = tree.nnodes

    A_ = np.zeros((n_tips, n_nodes))

    for i in np.arange(n_nodes):
        leaves_i = list(set(tree.get_node_descendant_idxs(i)) & set(np.arange(n_tips)))
        A_[leaves_i, i] = 1

    # collapsed trees may have scrambled leaves.
    # Therefore, we permute the rows of A such that they are in the original order. Columns (nodes) stay permuted.
    scrambled_leaves = list(tree.get_node_values("idx_orig", True, True)[-n_tips:])
    scrambled_leaves.reverse()
    if scrambled_leaves[0] == '':
        scrambled_leaves = list(np.arange(0, n_tips, 1))

    A = np.zeros((n_tips, n_nodes))
    for r in range(n_tips):
        A[scrambled_leaves[r], :] = A_[r, :]
    A = A[:, :-1]

    return A, n_nodes - 1


def collapse_singularities(
        tree: tt.tree
) -> tt.tree:
    """
    Collapses (deletes) nodes in a toytree tree that are singularities (have only one child).

    Parameters
    ----------
    tree
        A toytree tree object

    Returns
    -------
    A toytree tree without singularities

    tree_new
        A toytree tree
    """

    A, _ = get_A(tree)
    A_T = A.T
    unq, count = np.unique(A_T, axis=0, return_counts=True)

    repeated_idx = []
    for repeated_group in unq[count > 1]:
        repeated_idx.append(np.argwhere(np.all(A_T == repeated_group, axis=1)).ravel())

    nodes_to_delete = [i for idx in repeated_idx for i in idx[1:]]

    # _coords.update() scrambles the idx of leaves. Therefore, keep track of it here
    tree_new = tree.copy()
    for node in tree_new.treenode.traverse():
        node.add_feature("idx_orig", node.idx)

    for n in nodes_to_delete:
        node = tree_new.idx_dict[n]
        node.delete()

    tree_new._coords.update()

    # remove node artifacts
    for k in list(tree_new.idx_dict):
        if k >= tree_new.nnodes:
            tree_new.idx_dict.pop(k)

    return tree_new


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram
    """
    if i+1 < df_.shape[1]:
        a_inner = pd.unique(df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i+1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i+1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(
        df: pd.DataFrame,
        levels: List[str],
        inner_label: bool = True
) -> str:
    """
    Converts a pandas DataFrame with hierarchical information into a newick string.
    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram

    Parameters
    ----------
    df
        Pandas DataFrame that has one row for each leaf of the tree and columns that indicate a hierarchical ordering. See the tascCODA tutorial for an example.
    levels
        list that indicates how the columns in df are ordered as tree levels. Begins with the root level, ends with the leaf level
    inner_label
        Indicator whether labels for inner nodes should be included in the newick string

    Returns
    -------
    Newick string describing the tree structure from df

    newick
        A newick string
    """
    df_tax = df.loc[:, [x for x in levels if x in df.columns]]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick
