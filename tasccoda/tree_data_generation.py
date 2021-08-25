import toytree as tt
import numpy as np
import re

import sccoda.util.data_generation as gen
from tasccoda import tree_utils as util

# tree data generation functions (TODO: Remove)
def generate_tree_levels(K, D):

    a = np.arange(K).tolist()
    w_old = [len(a)]

    widths = [w_old]
    assignments = [a]

    for _ in np.arange(start=1, stop=D-1):

        w_new = np.random.binomial(w_old[0]-1, 0.3, size=1) + 1

        probs = np.repeat(1/w_new, w_new)
        a = np.ones(w_new, dtype=int) + np.random.multinomial(w_old - w_new, probs)

        assignments.append(a.tolist())
        widths.append(w_new.tolist())

        w_old = w_new

    ass_old = assignments[0]
    ass_new = assignments[0]

    for level in assignments[1:]:
        c = 0
        ass_new = []
        for nbranch in level:
            ass_new.append(ass_old[c:c+nbranch])

            c += nbranch
        ass_old = ass_new

    newick = str(ass_new)
    for square, round in {"[":"(", "]":")"}.items():
        newick = newick.replace(square, round)
    newick = re.sub(r"[0-9]+", "", newick)
    newick += ";"

    return newick


def effect_from_abs_changes(counts_before, counts_after, ref_index=0):

    if not counts_before.shape[0] == counts_after.shape[0]:
        raise ValueError("Before and after must have the same dimension!")

    # if (np.abs(np.sum(counts_before, axis=0) - np.sum(counts_after, axis=0)) >= 0.01*np.sum(counts_before, axis=0)):
    #     raise ValueError("Before and after must have the same sum!")

    n_total = np.sum(counts_before, axis=0)

    b_before = np.log(counts_before / n_total)
    b_after = np.log(counts_after / n_total)

    w = b_after - b_before
    w = w - w[ref_index]

    return b_before, w


def generate_random_tree_data(
        K: int,
        D: int,
        cells_per_type: int=1000,
        seed: int = None,
):

    if seed is not None:
        np.random.seed(seed)

    n_total = K * cells_per_type
    n_samples = [10, 10]

    newick = generate_tree_levels(K, D)
    data_tree = tt.tree(newick)
    # data_tree = util.collapse_singularities(data_tree)

    T = data_tree.nnodes

    # propose a tree-internal node as effect
    def propose_effect():
        effect_node = np.random.choice(np.arange(K, T))

        # if node has only one descendant, use the descendant instead (recursively)
        only_child = True
        while only_child:
            children = data_tree.idx_dict[effect_node].get_children()
            # if only one child, jump to it
            if len(children) == 1:
                if children[0].idx >= K:
                    effect_node = children[0].idx
                # if the only child is a leaf, restart (all nodes but the leaf will be deleted during collapse)
                else:
                    effect_node = np.random.choice(np.arange(K, T))
            else:
                only_child = False

        # get leaves of the effect node
        effect_leaves = [x for x in data_tree.get_node_descendant_idxs(effect_node) if x < K]

        return effect_node, effect_leaves

    effect_node, effect_leaves = propose_effect()

    c = 0
    while (c <= 20) and (len(effect_leaves) > K/3):
        effect_node, effect_leaves = propose_effect()
        c += 1

    while (c <= 100) and (len(effect_leaves) > K/2):
        effect_node, effect_leaves = propose_effect()
        c += 1


    counts_before = gen.counts_from_first(cells_per_type, n_total, K)
    try:
        counts_after = (np.repeat((n_total - len(effect_leaves) * 2 * cells_per_type) / (K - len(effect_leaves)), K)) + 1
    except ZeroDivisionError:
        counts_after = np.repeat(1, K)
    counts_after[effect_leaves] = 2 * cells_per_type

    b, w = effect_from_abs_changes(counts_before, counts_after, 9)

    test_data = gen.generate_case_control(
        1,
        K,
        n_total,
        n_samples,
        sigma=None,
        b_true=b,
        w_true=[w],
    )
    test_data.uns["phylo_tree"] = data_tree

    effect_info = [effect_node, effect_leaves]

    return test_data, effect_info


def generate_one_sample(N, mu, theta=99):
    p = theta*(mu/np.sum(mu))
    a = np.random.dirichlet(p)
    sample = np.random.multinomial(N, a)
    return sample


def generate_mu(a_abs, num_leaves, effect_nodes, effect_leaves, effect_size, newick):

    # base distribution
    alpha = np.random.uniform(-a_abs, a_abs, num_leaves)
    mu_0 = np.exp(alpha)

    # get A
    tree = tt.tree(newick)
    A, T = util.get_A(tree)

    eff = np.zeros(T)
    for n in effect_nodes:
        eff[n] = effect_size
    eff_l = np.matmul(A, eff)
    mu_1 = np.exp(alpha + eff_l)

    return mu_0, mu_1


def get_effect_nodes(newick, num_effects, num_leaves):
    # get A
    tree = tt.tree(newick)

    # get references
    ref_leaf = num_leaves - 1
    ref_nodes = [p.idx for p in tree.idx_dict[ref_leaf].get_ancestors()][:-1]
    ref_nodes = [ref_leaf] + ref_nodes

    # propose a node as effect
    def propose_effect(tree, ref_nodes):
        A, T = util.get_A(tree)
        P = A.shape[0]
        eff_node = np.random.choice(np.arange(0, T))

        only_child = True
        while only_child:
            # make sure node is not a reference node
            if eff_node in ref_nodes:
                eff_node = np.random.choice(np.arange(0, T))
                continue
            # if node is internal, make sure it is not a singularity (node with one child)
            if eff_node >= P:
                # if node has only one descendant, use the descendant instead (recursively)
                children = tree.idx_dict[eff_node].get_children()
                # if only one child, jump to it
                if len(children) == 1:
                    if children[0].idx >= P:
                        eff_node = children[0].idx
                    # if the only child is a leaf, restart (all nodes but the leaf will be deleted during collapse)
                    else:
                        eff_node = np.random.choice(np.arange(0, T))
                        continue
                else:
                    only_child = False
            else:
                only_child = False

        # get leaves of the effect node
        eff_leaves = [x for x in tree.get_node_descendant_idxs(eff_node) if x < P]
        return eff_node, eff_leaves

    effect_nodes = []
    effect_leaves = []
    for i in range(num_effects):
        n_, l_ = propose_effect(tree, ref_nodes)
        effect_nodes.append(n_)
        effect_leaves += l_

    effect_leaves = list(set(effect_leaves))

    print(f"effect_nodes: {effect_nodes}")
    print(f"effect_leaves: {effect_leaves}")

    return effect_nodes, effect_leaves
