"""
Results class that summarizes the results of tascCODA and calculates test statistics.
This class extends the ´´InferenceData`` class in the ``arviz`` package and can use all plotting and diacgnostic
functionalities of it.

Additionally, this class can produce nicely readable outputs for scCODA.

:authors: Johannes Ostner
"""
import numpy as np
import arviz as az
import pandas as pd
import tasccoda.tree_utils as util

from typing import Optional, Tuple, Collection, Union, List


class CAResultConverter_tree(az.data.io_dict.DictConverter):
    """
    Helper class for result conversion into arviz's format
    """

    def to_result_data(self, sampling_stats, model_specs):

        post = self.posterior_to_xarray()
        ss = self.sample_stats_to_xarray()
        postp = self.posterior_predictive_to_xarray()
        prior = self.prior_to_xarray()
        ssp = self.sample_stats_prior_to_xarray()
        prip = self.prior_predictive_to_xarray()
        obs = self.observed_data_to_xarray()

        return CAResult_tree(
            sampling_stats=sampling_stats,
            model_specs=model_specs,
            **{
                "posterior": post,
                "sample_stats": ss,
                "posterior_predictive": postp,
                "prior": prior,
                "sample_stats_prior": ssp,
                "prior_predictive": prip,
                "observed_data": obs,
            }
        )

class CAResult_tree(az.InferenceData):
    """
    Result class for tascCODA, extends the arviz framework for inference data.

    The CAResult_tree class is an extension of az.InferenceData, that adds some information about the compositional model
    and is able to print humanly readable results.
    It supports all functionality from az.InferenceData.
    """


    def __init__(
            self,
            sampling_stats: dict,
            model_specs: dict,
            **kwargs
    ):
        """
        Gathers sampling information from a compositional model and converts it to a ``az.InferenceData`` object.
        The following attributes are added during class initialization:

        ``self.sampling_stats``: dict - see below
        ``self.model_specs``: dict - see below

        ``self.intercept_df``: Intercept dataframe from ``CAResult.summary_prepare``
        ``self.effect_df``: Effect dataframe from ``CAResult.summary_prepare``

        Parameters
        ----------
        sampling_stats
            Information and statistics about the MCMC sampling procedure.
            Default keys:
            - "chain_length": Length of MCMC chain (with burnin samples)
            - "num_burnin": Number of burnin samples
            - "acc_rate": MCMC Acceptance rate
            - "duration": Duration of MCMC sampling

        model_specs
            All information and statistics about the model specifications.
            Default keys:
            - "formula": Formula string
            - "reference": int - identifier of reference cell type

            Added during class initialization:
            - "threshold_prob": Threshold for inclusion probability that separates significant from non-significant effects
        kwargs
            passed to az.InferenceData. This includes the MCMC chain states and statistics for eachs MCMC sample.
        """
        super(self.__class__, self).__init__(**kwargs)

        self.sampling_stats = sampling_stats
        self.model_specs = model_specs
        self.is_spikeslab = self.model_specs["spike_slab"]

        intercept_df, effect_df, node_df = self.summary_prepare()

        self.intercept_df = intercept_df
        self.effect_df = effect_df
        self.node_df = node_df

    def summary_prepare(
            self,
            *args,
            **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates summary dataframes for intercepts and slopes.
        This function builds on and supports all functionalities from ``az.summary``.

        Parameters
        ----------
        args
            Passed to ``az.summary``
        kwargs
            Passed to ``az.summary``

        Returns
        -------
        Intercept and effect DataFrames

        intercept_df -- pandas df
            Summary of intercept parameters. Contains one row per cell type.

            Columns:
            - Final Parameter: Final intercept model parameter
            - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            - SD: Standard deviation of MCMC samples
            - Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

        effect_df -- pandas df
            Summary of effect (slope) parameters. Contains one row per covariate/cell type combination.

            Columns:
            - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
            - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            - SD: Standard deviation of MCMC samples
            - Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
            - log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
            - Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.
        """

        # initialize summary df from arviz and separate into intercepts and effects.

        func_dict = {
            "median": np.median
        }
        summ = az.summary(self, *args, **kwargs,
                          kind="stats", var_names=["alpha", "beta_select", "beta"], stat_funcs=func_dict)
        effect_df = summ.loc[summ.index.str.match("|".join(["beta\["]))].copy()
        intercept_df = summ.loc[summ.index.str.match("|".join(["alpha\["]))].copy()
        node_df = summ.loc[summ.index.str.match("|".join(["beta_select\["]))].copy()

        # Build neat index
        cell_types = self.posterior.coords["cell_type"].values
        covariates = self.posterior.coords["covariate"].values
        cell_types_node = self.posterior.coords["cell_type_select"].values
        covariates_node = [x + "_node" for x in covariates]

        intercept_df.index = pd.Index(cell_types, name="Cell Type")
        effect_df.index = pd.MultiIndex.from_product([covariates, cell_types],
                                                     names=["Covariate", "Cell Type"])
        node_df.index = pd.MultiIndex.from_product([covariates_node, cell_types_node],
                                                     names=["Covariate", "Node"])

        # Calculation of columns that are not from az.summary
        node_df = self.complete_node_df(node_df)
        intercept_df = self.complete_alpha_df(intercept_df)
        effect_df = self.complete_beta_df(intercept_df, effect_df, node_df)

        # Give nice column names, remove unnecessary columns
        hdis = intercept_df.columns[intercept_df.columns.str.contains("hdi")]
        hdis_new = hdis.str.replace("hdi_", "HDI ")

        intercept_df = intercept_df.loc[:, ["final_parameter", hdis[0], hdis[1], "sd", "expected_sample"]].copy()
        intercept_df = intercept_df.rename(columns=dict(zip(
            intercept_df.columns,
            ["Final Parameter", hdis_new[0], hdis_new[1], "SD", "Expected Sample"]
        )))

        effect_df = effect_df.loc[:, ["final_parameter", "median", hdis[0], hdis[1], "sd",
                                      "expected_sample", "log_fold"]].copy()
        effect_df = effect_df.rename(columns=dict(zip(
            effect_df.columns,
            ["Effect", "Median", hdis_new[0], hdis_new[1], "SD",
             "Expected Sample", "log2-fold change"]
        )))

        node_df = node_df.loc[:, ["final_parameter", "median", hdis[0], hdis[1], "sd", "delta", "significant"]].copy()
        node_df = node_df.rename(columns=dict(zip(
            node_df.columns,
            ["Final Parameter", "Median", hdis_new[0], hdis_new[1], "SD", "Delta", "Is significant"]
        )))
        return intercept_df, effect_df, node_df

    def complete_beta_df(
            self,
            intercept_df: pd.DataFrame,
            effect_df: pd.DataFrame,
            node_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluation of MCMC results for effect parameters. This function is only used within self.summary_prepare.
        This function also calculates the posterior inclusion probability for each effect and decides whether effects are significant.

        Parameters
        ----------
        intercept_df
            Intercept summary, see ``self.summary_prepare``
        effect_df
            Effect summary, see ``self.summary_prepare``

        Returns
        -------
        effect DataFrame

        effect_df
            DataFrame with inclusion probability, final parameters, expected sample
        """
        beta_inc_prob = []
        beta_nonzero_mean = []

        # Get effects of nodes on leaves
        D = len(effect_df.index.levels[0])
        effect_df["final_parameter"] = np.matmul(np.kron(np.eye(D,dtype=int), self.model_specs["A"]), np.array(node_df["final_parameter"]))

        # Get expected sample, log-fold change
        K = len(effect_df.index.levels[1])

        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alpha_par = intercept_df.loc[:, "final_parameter"]
        alphas_exp = np.exp(alpha_par)
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values

        beta_mean = alpha_par
        beta_sample = []
        log_sample = []

        for d in range(D):
            beta_d = effect_df.loc[:, "final_parameter"].values[(d*K):((d+1)*K)]
            beta_d = (beta_mean + beta_d)
            beta_d = np.exp(beta_d)
            beta_d = beta_d / np.sum(beta_d) * y_bar

            beta_sample = np.append(beta_sample, beta_d)
            log_sample = np.append(log_sample, np.log2(beta_d/alpha_sample))

        effect_df.loc[:, "expected_sample"] = beta_sample
        effect_df.loc[:, "log_fold"] = log_sample

        return effect_df

    def complete_alpha_df(
            self,
            intercept_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluation of MCMC results for intercepts. This function is only used within self.summary_prepare.

        Parameters
        ----------
        intercept_df
            Intercept summary, see self.summary_prepare

        Returns
        -------
        intercept DataFrame

        intercept_df
            Summary DataFrame with expected sample, final parameters
        """

        intercept_df = intercept_df.rename(columns={"mean": "final_parameter"})

        # Get expected sample
        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alphas_exp = np.exp(intercept_df.loc[:, "final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        intercept_df.loc[:, "expected_sample"] = alpha_sample

        return intercept_df

    def complete_node_df(
            self,
            node_df: pd.DataFrame,
            target_fdr: float = 0.05,
    ) -> pd.DataFrame:

        # calculate inclusion threshold
        theta = np.median(self.posterior["theta"].values)
        l_0 = self.model_specs["lambda_0"]
        l_1 = self.model_specs["lambda_1"]

        def delta(l_0, l_1, theta):
            p_t = (theta * l_1 / 2) / ((theta * l_1 / 2) + ((1 - theta) * l_0 / 2))
            return 1 / (l_0 - l_1) * np.log(1 / p_t - 1)

        D = len(node_df.index.levels[0])

        # apply inclusion threshold
        deltas = delta(l_0, l_1, theta)
        for ct in self.model_specs["reference_nodes"]:
            deltas = np.concatenate([
                    deltas[:ct],
                    [0],
                    deltas[ct:]
                ])
        node_df["delta"] = np.tile(deltas, D)
        node_df["significant"] = np.abs(node_df["median"]) > node_df["delta"]
        node_df["final_parameter"] = np.where(
            node_df.loc[:, "significant"] == True,
            node_df.loc[:, "median"],
            0)

        return node_df

    def summary(
            self,
            *args,
            **kwargs
    ):
        """
        Printing method for scCODA's summary.

        Usage: ``result.summary()``

        Parameters
        ----------
        args
            Passed to az.summary
        kwargs
            Passed to az.summary

        Returns
        -------
        prints to console

        """

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            intercept_df, effect_df, node_df = self.summary_prepare(*args, **kwargs)
        else:
            intercept_df = self.intercept_df
            effect_df = self.effect_df
            node_df = self.node_df

        # Get number of samples, cell types
        if self.sampling_stats["y_hat"] is not None:
            data_dims = self.sampling_stats["y_hat"].shape
        else:
            data_dims = (10, 5)

        # Cut down DataFrames to relevant info
        alphas_print = intercept_df.loc[:, ["Final Parameter", "Expected Sample"]]
        betas_print = effect_df.loc[:, ["Effect", "Expected Sample", "log2-fold change"]]
        node_print = node_df.loc[:, ["Final parameter", "Is significant"]]

        # Print everything neatly
        print("Compositional Analysis summary:")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Reference index: %s" % str(self.model_specs["reference"]))
        print("Formula: %s" % self.model_specs["formula"])
        print("")
        print("Intercepts:")
        print(alphas_print)
        print("")
        print("")
        print("Effects:")
        print(betas_print)
        print("")
        print("")
        print("Nodes:")
        print(node_print)

    def summary_extended(
            self,
            *args,
            **kwargs
    ):

        """
        Extended (diagnostic) printing function that shows more info about the sampling result

        Parameters
        ----------
        args
            Passed to az.summary
        kwargs
            Passed to az.summary

        Returns
        -------
        Prints to console

        """

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            intercept_df, effect_df, node_df = self.summary_prepare(*args, **kwargs)
        else:
            intercept_df = self.intercept_df
            effect_df = self.effect_df
            node_df = self.node_df

        # Get number of samples, cell types
        data_dims = self.sampling_stats["y_hat"].shape

        # Print everything
        print("Compositional Analysis summary (extended):")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Reference index: %s" % str(self.model_specs["reference"]))
        print("Formula: %s" % self.model_specs["formula"])

        print("MCMC Sampling: Sampled {num_results} chain states ({num_burnin} burnin samples) in {duration:.3f} sec. "
              "Acceptance rate: {ar:.1f}%".format(num_results=self.sampling_stats["chain_length"],
                                                  num_burnin=self.sampling_stats["num_burnin"],
                                                  duration=self.sampling_stats["duration"],
                                                  ar=(100*self.sampling_stats["acc_rate"])))
        print("")
        print("Intercepts:")
        print(intercept_df)
        print("")
        print("")
        print("Effects:")
        print(effect_df)
        print("")
        print("")
        print("Nodes:")
        print(node_df)

    def get_significant_results(self, *args, **kwargs):

        if args or kwargs:
            intercept_df, effect_df, node_df = self.summary_prepare(*args, **kwargs)
        else:
            intercept_df = self.intercept_df
            effect_df = self.effect_df
            node_df = self.node_df

        sig_nodes = node_df[node_df["Final Parameter"] != 0].index.get_level_values(1).tolist()
        res_otus = np.where(np.matmul(self.model_specs["A"], self.node_df["Final Parameter"]) != 0)[0].tolist()
        res_otus = [str(x) for x in res_otus]

        return {
            "Nodes": sig_nodes,
            "Cell types": res_otus
        }

    def draw_tree_effects(
            self,
            tree,
            *args,
            **kwargs
    ):

        tree2 = util.collapse_singularities(tree)

        effs = self.node_df.copy()
        effs.index = effs.index.get_level_values("Node")

        for n in tree2.treenode.traverse():
            if n.name in effs.index:
                e = effs.loc[n.name, "Final Parameter"]
                n.add_feature("effect", e)
            else:
                n.add_feature("effect", 0)

        # add node colors
        for n in tree2.treenode.traverse():
            if np.sign(n.effect) == 1:
                n.add_feature("color", "black")
            elif np.sign(n.effect) == -1:
                n.add_feature("color", "white")
            else:
                n.add_feature("color", "cyan")

        eff_max = np.max([np.abs(n.effect) for n in tree2.treenode.traverse()])
        if eff_max > 0:
            ns = [(np.abs(x) * 20 / eff_max) + 5 if x != 0 else 0 for x in tree2.get_node_values("effect", 1, 1)]
        else:
            ns = 0

        tree2.draw(
            node_sizes=ns,
            node_colors=tree2.get_node_values("color", 1, 1),
            node_style={
                "stroke": "black",
                "stroke-width": "1"
            },
            *args,
            **kwargs
        )
