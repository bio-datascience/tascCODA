"""
Initialization of tascCODA models.

:authors: Johannes Ostner
"""
import numpy as np
import patsy as pt

from anndata import AnnData
from typing import Union, Optional, Tuple

from tasccoda import tree_agg_model_sslasso as ssl

from tasccoda import tree_utils as util


class CompositionalAnalysisTree:
    """
    Initializer class for tascCODA models. This class is called when performing compositional analysis with tascCODA.

    Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", reference_cell_type="CellTypeA", pen_args={"phi": 0})

    Calling an scCODA model requires these parameters:

    data
        anndata object with cell counts as data.X and covariates saved in data.obs
    formula
        patsy-style formula for building the covariate matrix.
        Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
        To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
    reference_cell_type
        Column index that sets the reference cell type. Can either reference the name of a column or a column number (starting at 0).
        If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
    pen_args
        Dictionary with penalty arguments. The parameters phi (aggregation bias), lambda_1, lambda_0 can be set here.
        See the tascCODA paper for an explanation of these parameters. Default: lambda_0 = 50, lambda_1 = 5, phi = 0.
    """

    def __new__(
            cls,
            data: AnnData,
            formula: str,
            reference_cell_type: Union[str, int] = "automatic",
            reg: str = "scaled_3",
            pen_args: dict = {"lambda": 5},
            automatic_reference_absence_threshold: float = 0.05,
            *args,
            **kwargs
    ) -> Union[ssl.TreeModelSSLasso]:
        """
        Builds count and covariate matrix, returns a CompositionalModel object

        Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", reference_cell_type="CellTypeA")

        Parameters
        ----------
        data
            anndata object with cell counts as data.X and covariates saved in data.obs.
            The tree structure must be saved as a toytree.tree object in data.uns["phylo_tree"].
            See the tutorial on how to generate the tree structure.
        formula
            R-style formula for building the covariate matrix.
            Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
            To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
        reference_cell_type
            Column index that sets the reference cell type. Can either reference the name of a column or the n-th column (indexed at 0).
            If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
        reg
            Indicator for regularization scheme. Default: "scaled_3". Other schemes are only available for legacy reasons and should not be used.
        pen_args
            Dictionary with penalty arguments. With `reg="scaled_3"`, the parameters phi (aggregation bias), lambda_1, lambda_0 can be set here.
            See the tascCODA paper for an explanation of these parameters. Default: lambda_0 = 50, lambda_1 = 5, phi = 0.
        automatic_reference_absence_threshold
            If using reference_cell_type = "automatic", determine what the maximum fraction of zero entries for a cell type is to be considered as a possible reference cell type

        Returns
        -------
        A compositional model

        model
            A tascCODA.tree_agg_model_sslasso.TreeModelSSLasso object
        """

        # Collapse singularities in the tree
        phy_tree = util.collapse_singularities(data.uns["phylo_tree"])

        # Get ancestor matrix
        A, T = util.get_A(phy_tree)

        # Bring names of nodes back in order, they might have been scrambled during collapse_singularities
        node_names = [n.name for n in phy_tree.idx_dict.values()][1:]
        node_names.reverse()

        order = [n.name for n in data.uns["phylo_tree"].treenode.traverse() if n.is_leaf()]
        order.reverse()
        order_ind = [data.var.index.tolist().index(x) for x in order]

        var2 = data.var.reindex(order)
        X2 = data.X[:, order_ind]
        cell_types = var2.index.to_list()

        # Get count data
        data_matrix = X2.astype("float64")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Automatic reference selection (dispersion-based)
        if reference_cell_type == "automatic":
            percent_zero = np.sum(data_matrix == 0, axis=0)/data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < automatic_reference_absence_threshold)[0]

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            # select reference
            cell_type_disp = np.var(rel_abun, axis=0)/np.mean(rel_abun, axis=0)
            min_var = np.min(cell_type_disp[nonrare_ct])
            ref_index = np.where(cell_type_disp == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection! Reference cell type set to {ref_cell_type}")

            node_ind = phy_tree.get_mrca_idx_from_tip_labels(ref_cell_type)

        # Column name as reference cell type
        elif reference_cell_type in cell_types:
            node_ind = phy_tree.get_mrca_idx_from_tip_labels(reference_cell_type)

        # Numeric reference cell type
        elif isinstance(reference_cell_type, int) & (reference_cell_type < len(cell_types)) & (reference_cell_type >= 0):
            node_ind = reference_cell_type

        # None of the above: Throw error
        else:
            raise NameError("Reference index is not a valid cell type name or numerical index!")

        # Ancestors of reference are a reference, too!
        refs = [p.idx for p in phy_tree.idx_dict[node_ind].get_ancestors()][:-1]
        refs = [node_ind] + refs

        # number of leaves for each internal node (important for aggregation penalty lambda_1)
        if "node_leaves" not in pen_args:
            node_leaves = [len(n.get_leaves()) for n in phy_tree.idx_dict.values()]
            node_leaves.reverse()
            pen_args["node_leaves"] = np.delete(np.array(node_leaves[:-1]), refs)

        # Default spike-and-slab LASSO parameters
        if "lambda_0" not in pen_args:
            pen_args["lambda_0"] = 50
        if "lambda_1" not in pen_args:
            pen_args["lambda_1"] = 5
        if "phi" not in pen_args:
            pen_args["phi"] = 0

        # Generate model object
        return ssl.TreeModelSSLasso(
            covariate_matrix=np.array(covariate_matrix),
            data_matrix=data_matrix,
            cell_types=cell_types,
            node_names=node_names,
            covariate_names=covariate_names,
            reference_nodes=refs,
            formula=formula,
            A=A,
            T=T,
            reg=reg,
            pen_args=pen_args,
            *args,
            **kwargs
        )