# tascCODA
Tree-aggregated compositional analysis for high-throughput sequencing data

tascCODA extends the [scCODA model](https://github.com/theislab/scCODA) (Büttner, Ostner et al., 2021)
with a method to perform sparse, tree-aggregated modeling of high-throughput sequencing data.


The statistical methodology and benchmarking performance are described in:
 
TODO

Code for reproducing the analysis from the paper is available [here](https://github.com/bio-datascience/tascCODA_reproducibility).

## Installation

Running the package requires a working Python environment (>=3.8).

This package uses the `tensorflow` (`>=2.4`) and `tensorflow-probability` (`>=0.12`) packages.
The GPU computation features of these packages have not been tested with tascCODA and are thus not recommended.
    
**To install tascCODA via pip, call**:

    pip install tasccoda


**To install tascCODA from source**:

- Navigate to the directory that you want to install tascCODA in
- Clone the repository from [Github](https://github.com/bio-datascience/tascCODA):

    `git clone https://github.com/bio-datascience/tascCODA`

- Navigate to the root directory of tascCODA:

    `cd tascCODA`

- Install dependencies::

    `pip install -r requirements.txt`

- Install the package:

    `python setup.py install`


## Usage

Import tascCODA in a Python session via:

    import tasccoda

You can then import a dataset in the same way as scCODA (see [here](https://sccoda.readthedocs.io/en/latest/) for an instruction on scCODA's data structure)
Once you imported your dataset, add a [toytree](https://github.com/eaton-lab/toytree) tree object, for example generated from a Newick string, as `data.uns["phylo_tree"]`.

Then, initialize your analysis object, together with your formula and reference feature (see the scCODA documentation for explanations).
To set the aggregation bias, pass `"phi"` as a key in the `pen_args` parameter

    `model = tasccoda.tree_ana.CompositionalAnalysisTree(
    data,
    reference_cell_type="9",
    formula="x_0",
    pen_args={"phi":phi},
    )
    `

Then, run HMC sampling with dual-averaging step size adaptation by calling:

    `result = model.sample_hmc_da()`

Finally, you can look at `result.node_df` to find credible effects of covariates on tree nodes 
or plot a tree with indicators for credible effects with `result.draw_tree_effects()`