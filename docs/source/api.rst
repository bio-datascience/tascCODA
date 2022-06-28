.. automodule:: tasccoda

API
===

We advise to import tascCODA in a python session via::

    import tasccoda
    ana = tasccoda.tree_ana

The workflow in tascCODA starts with reading in HTS count data and visualizing them through scCODA (``sccoda.util.data_visualization``).
Additionally a tree structure has to be generated from hierarchical information and added to the data.

Data acquisition
----------------

**Integrating data sources (dat)**

Data integration works just as in scCODA. The tree structure must be added manually (see tree-structured data)

**Tree data handling utilities**

.. autosummary::
    :toctree: .

    tasccoda.tree_utils.get_A
    tasccoda.tree_utils.collapse_singularities
    tasccoda.tree_utils.df2newick

**Compositional data visualization**

Compositional datasets can be plotted via the methods from scCODA (``sccoda.util.data_visualization``).

Model setup and inference
-------------------------

Using the tascCODA model is easiest by generating an instance of ``ana.CompositionalAnalysisTree``.
By specifying the formula via the `patsy <https://patsy.readthedocs.io/en/latest/>`_ syntax, many combinations and
transformations of the covariates can be performed without redefining the covariate matrix.
Also, the reference feature and the tree aggregation bias ``phi`` need to be specified in this step.

**The tascCODA model**

.. autosummary::
    :toctree: .

    tasccoda.tree_ana.CompositionalAnalysisTree
    tasccoda.tree_agg_model_sslasso.TreeModelSSLasso


Result evaluation
-----------------

Executing an inference method on a compositional model produces a ``sccoda.util.result_classes.CAResult`` object. This
class extends the ``InferenceData`` class of `arviz <https://arviz-devs.github.io/arviz/>`_ and supports all its
diagnostic and plotting functionality. Furthermore, a function to plot the tree structure with effect labels is included.

.. autosummary::
    :toctree: .

    tasccoda.tree_results.CAResult_tree
