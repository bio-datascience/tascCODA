
import numpy as np
import warnings

import tensorflow as tf
import tensorflow_probability as tfp

from tasccoda import tree_results as res
from sccoda.model import scCODA_model as mod
from typing import Optional, Tuple, Collection, Union, List

tfd = tfp.distributions
tfb = tfp.bijectors


class TreeModelSSLasso(mod.CompositionalModel):
    """
    Statistical model for single-cell differential composition analysis with specification of a reference cell type.
    This is the standard scCODA model and recommenced for all uses.

    The hierarchical formulation of the model for one sample is:

    .. math::
         y|x &\\sim DirMult(a(x), \\bar{y}) \\\\
         \\log(a(x)) &= \\alpha + x \\beta \\\\
         \\alpha_k &\\sim N(0, 5) \\quad &\\forall k \\in [K] \\\\
         \\beta_{d, \\hat{k}} &= 0 &\\forall d \\in [D]\\\\
         \\beta_{d, k} &= \\tau_{d, k} \\tilde{\\beta}_{d, k} \\quad &\\forall d \\in [D], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tau_{d, k} &= \\frac{\\exp(t_{d, k})}{1+ \\exp(t_{d, k})} \\quad &\\forall d \\in [D], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\frac{t_{d, k}}{50} &\\sim N(0, 1) \\quad &\\forall d \\in [D], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tilde{\\beta}_{d, k} &= (\\tilde{\\mu} + \\tilde{\\sigma}^2) \\cdot \\tilde{\\gamma}_{d, k} \\quad &\\forall d \\in [D], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tilde{\\mu} &\\sim N(0, 1) \\\\
         \\tilde{\\sigma}^2 &\\sim HC(0, 1) \\\\
         \\tilde{\\gamma}_{d, k} &\\sim N(0,1) \\quad &\\forall d \\in [D], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\

    with y being the cell counts and x the covariates.

    For further information, see `scCODA: A Bayesian model for compositional single-cell data analysis`
    (Büttner, Ostner et al., 2020)

    """

    def __init__(
            self,
            node_names: List[str],
            reference_nodes: List[int],
            A: np.ndarray,
            T: int,
            reg: str = "scaled_3",
            pen_args: dict = {"lambda": 5},
            *args,
            **kwargs):
        """
        Constructor of model class. Defines model structure, log-probability function, parameter names,
        and MCMC starting values.

        Parameters
        ----------
        reference_cell_type
            Index of reference cell type (column in count data matrix)
        args
            arguments passed to top-level class
        kwargs
            arguments passed to top-level class
        """

        super(self.__class__, self).__init__(*args, **kwargs)

        self.node_names = node_names
        self.reference_nodes = reference_nodes
        self.reference_cell_type = reference_nodes[0]
        self.n_ref_nodes = len(reference_nodes)
        self.A = A
        self.T = T
        self.reg = reg
        self.pen_args = pen_args
        dtype = tf.float64

        # different penalty scalings (for legacy reasons. Default is "scaled_3")
        if reg == "scaled" or reg == "scaled_2":
            if self.pen_args["phi"] >= 0:
                self.penalty_scale_factor = tf.cast(((self.pen_args["node_leaves"]/self.K)**self.pen_args["phi"]), dtype)
            else:
                self.penalty_scale_factor = tf.cast((self.pen_args["node_leaves"].astype("float")**self.pen_args["phi"]), dtype)

            if reg == "scaled_2":
                lambda_1 = self.pen_args["lambda_1"] * self.penalty_scale_factor + 0.1
                lambda_0 = self.pen_args["lambda_0"]
            else:
                lambda_1 = self.pen_args["lambda_1"]
                lambda_0 = self.pen_args["lambda_0"] * self.penalty_scale_factor

        elif reg == "scaled_3":
            lambda_0 = self.pen_args["lambda_0"]
            self.penalty_scale_factor = tf.cast((1/(1+np.exp(-1*self.pen_args["phi"]*(self.pen_args["node_leaves"]/self.K-0.5)))), dtype)
            lambda_1 = 2 * self.pen_args["lambda_1"] * self.penalty_scale_factor

        else:
            lambda_0 = self.pen_args["lambda_0"]
            lambda_1 = self.pen_args["lambda_1"]

        self.l_0 = lambda_0
        self.l_1 = lambda_1

        # All parameters that are returned for analysis
        self.param_names = [
            "alpha_0",
            "b_0",
            "alpha_1",
            "b_1",
            "theta",
            "alpha",
            "bet_0",
            "bet_1",
            "beta_select",
            "beta",
            "concentration",
            "prediction"
        ]

        alpha_size = [self.K]
        beta_nobl_size = [self.D, self.T-self.n_ref_nodes]

        d = self.D * (self.T-self.n_ref_nodes)

        Root = tfd.JointDistributionCoroutine.Root

        def model():

            alpha_0 = yield Root(tfd.Independent(
                tfd.Exponential(
                    rate=(lambda_0 ** 2) / 2 * tf.ones(beta_nobl_size, dtype=dtype),
                    name="alpha_0"),
                reinterpreted_batch_ndims=2))

            b_raw_0 = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(beta_nobl_size, dtype=dtype),
                    scale=tf.ones(beta_nobl_size, dtype=dtype),
                    name="b_raw_0"),
                reinterpreted_batch_ndims=2))

            b_tilde_0 = alpha_0 * b_raw_0

            alpha_1 = yield Root(tfd.Independent(
                tfd.Exponential(
                    rate=(lambda_1 ** 2) / 2 * tf.ones(beta_nobl_size, dtype=dtype),
                    name="alpha_1"),
                reinterpreted_batch_ndims=2))

            b_raw_1 = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(beta_nobl_size, dtype=dtype),
                    scale=tf.ones(beta_nobl_size, dtype=dtype),
                    name="b_raw_1"),
                reinterpreted_batch_ndims=2))

            b_tilde_1 = alpha_1 * b_raw_1

            # Spike-and-slab
            theta = yield Root(tfd.Independent(
                tfd.Beta(
                    concentration1=tf.ones(1, dtype=dtype),
                    concentration0=tf.ones(1, dtype=dtype)*d,
                    name="theta"),
                reinterpreted_batch_ndims=1))

            # calculate proposed beta and perform spike-and-slab
            b_tilde = (1. - tf.cast(theta, dtype=dtype)) * b_tilde_0 + tf.cast(theta, dtype=dtype) * b_tilde_1

            # Include effect 0 for reference cell type
            for ct in self.reference_nodes:
                b_tilde = tf.concat(
                    axis=1, values=[
                        b_tilde[:, :ct],
                        tf.zeros(shape=[self.D, 1], dtype=dtype),
                        b_tilde[:, ct:]
                    ])

            # sum up tree levels
            beta = tf.matmul(b_tilde, A, transpose_b=True)

            # Intercepts

            alpha = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size, dtype=dtype),
                    scale=tf.ones(alpha_size, dtype=dtype)*10,
                    name="alpha"),
                reinterpreted_batch_ndims=1))

            # log-link function
            concentrations = tf.exp(alpha + tf.matmul(self.x, beta))

            # Cell count prediction via DirMult
            predictions = yield Root(tfd.Independent(
                tfd.DirichletMultinomial(
                    total_count=tf.cast(self.n_total, dtype),
                    concentration=concentrations,
                    name="predictions"
                ),
                reinterpreted_batch_ndims=1))

        self.model_struct = tfd.JointDistributionCoroutine(model)

        @tf.function(experimental_compile=True, autograph=False)
        def target_log_prob_fn(*args):
            log_prob = self.model_struct.log_prob(list(args) + [tf.cast(self.y, dtype)])
            return log_prob

        self.target_log_prob_fn = target_log_prob_fn

        # MCMC starting values
        self.init_params = [
            tf.ones(beta_nobl_size, name="init_a_0", dtype=dtype) * 1/lambda_0,
            tf.random.normal(beta_nobl_size, 0, 1, name='init_b_0', dtype=dtype),
            tf.ones(beta_nobl_size, name="init_a_1", dtype=dtype) * 1/lambda_1,
            tf.random.normal(beta_nobl_size, 0, 1, name='init_b_1', dtype=dtype),
            tf.ones(1, name="init_theta", dtype=dtype) * 0.5,
            tf.random.normal(shape=alpha_size, mean=0, stddev=1, name='init_alpha', dtype=dtype)
        ]

        # bijectors
        self.constraining_bijectors = [
            tfb.Exp(),
            tfb.Identity(),
            tfb.Exp(),
            tfb.Identity(),
            tfb.Sigmoid(),
            tfb.Identity(),
        ]


    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(
            self,
            states_burnin: List[any],
            num_results: int,
            num_burnin: int,
    ) -> np.ndarray:
        """
        Calculate posterior mode of cell counts (for analysis purposes) and add intermediate parameters
        that are no priors to MCMC results.

        Parameters
        ----------
        states_burnin
            MCMC chain without burn-in samples
        num_results
            Chain length (with burn-in)
        num_burnin
            Number of burn-in samples

        Returns
        -------
        posterior mode

        y_mean
            posterior mode of cell counts
        """

        chain_size_y = [num_results - num_burnin, self.N, self.K]
        chain_size_beta = [num_results - num_burnin, self.D, self.T]

        alphas = states_burnin[5]
        alphas_final = alphas.mean(axis=0)

        alpha_0 = states_burnin[0]
        b_0 = states_burnin[1]
        alpha_1 = states_burnin[2]
        b_1 = states_burnin[3]
        thetas = states_burnin[4]

        beta_0 = np.einsum("..., ...", alpha_0, b_0)
        beta_1 = np.einsum("..., ...", alpha_1, b_1)

        beta_select = np.einsum("...i, ...jk->...jk", 1-thetas, beta_0) + np.einsum("...i, ...jk->...jk", thetas, beta_1)

        for ct in self.reference_nodes:

            beta_select = np.concatenate([beta_select[:, :, :ct],
                                       np.zeros(shape=[num_results - num_burnin, self.D, 1], dtype=np.float64),
                                       beta_select[:, :, ct:]], axis=2)


        beta_ = np.matmul(beta_select, np.transpose(self.A))

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - num_burnin, 1, self.K)))

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results - num_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(beta_0)
        states_burnin.append(beta_1)
        states_burnin.append(beta_select)
        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float64)

        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]

        return y_mean

    def sample_hmc(
            self,
            num_results: int = int(20e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            num_leapfrog_steps: Optional[int] = 10,
            step_size: float = 0.01
    ) -> res.CAResult_tree:

        """
        Hamiltonian Monte Carlo (HMC) sampling in tensorflow 2.

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        - `step_size`: The step size used by the algorithm in each step

        Parameters
        ----------
        num_results
            MCMC chain length (default 20000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        num_leapfrog_steps
            HMC leapfrog steps (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        results object

        result
            Compositional analysis result
        """

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=self.constraining_bijectors)

        # Set default value for adaptation steps if none given
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Add step size adaptation (Andrieu, Thomas - 2008)
        hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.75)

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        hmc_kernel = tfp.experimental.mcmc.WithReductions(hmc_kernel, pbar)

        # diagnostics tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.inner_results.log_accept_ratio < -1000.),
                'is_accepted': pkr.inner_results.inner_results.inner_results.is_accepted,
                'step_size': pkr.inner_results.inner_results.inner_results.accepted_results.step_size,
            }

        # The actual HMC sampling process
        states, kernel_results, duration = self.sampling(num_results, num_burnin,
                                                         hmc_kernel, self.init_params, trace_fn)
        pbar.bar.close()
        # apply burn-in
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=False)

        # Calculate posterior predictive
        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {
            "chain_length": num_results,
            "num_burnin": num_burnin,
            "acc_rate": acc_rate,
            "duration": duration,
            "y_hat": y_hat
        }

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def sample_hmc_da(
            self,
            num_results: int = int(20e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            num_leapfrog_steps: Optional[int] = 10,
            step_size: float = 0.01
    ) -> res.CAResultConverter_tree:
        """
        HMC sampling with dual-averaging step size adaptation (Nesterov, 2009)

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `log_acc_ratio`: log-acceptance ratio

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        - `step_size`: The step size used by the algorithm in each step

        Parameters
        ----------
        num_results
            MCMC chain length (default 20000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        num_leapfrog_steps
            HMC leapfrog steps (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        result object

        result
            Compositional analysis result
        """

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=self.constraining_bijectors)

        # Set default value for adaptation steps if none given
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Add step size adaptation
        hmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.85, decay_rate=0.75)

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        hmc_kernel = tfp.experimental.mcmc.WithReductions(hmc_kernel, pbar)

        # tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.inner_results.log_accept_ratio < -1000.),
                "log_acc_ratio": pkr.inner_results.inner_results.inner_results.log_accept_ratio,
                'is_accepted': pkr.inner_results.inner_results.inner_results.is_accepted,
                'step_size': tf.exp(pkr.inner_results.log_averaging_step[0]),
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, num_burnin, hmc_kernel, self.init_params, trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=False)
        pbar.bar.close()

        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {
            "chain_length": num_results,
            "num_burnin": num_burnin,
            "acc_rate": acc_rate,
            "duration": duration,
            "y_hat": y_hat
        }

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def sample_nuts(
            self,
            num_results: int = int(10e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            max_tree_depth: int = 10,
            step_size: float = 0.01
    ) -> res.CAResult_tree:
        """
        HMC with No-U-turn (NUTS) sampling.
        This method is untested and might yield different results than expected.

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `leapfrogs_taken`: Number of leapfrog steps taken by the integrator

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `energy`: HMC "Energy" value for each step

        - `log_accept_ratio`: log-acceptance ratio

        - `step_size`: The step size used by the algorithm in each step

        - `reached_max_depth`: Whether the NUTS algorithm reached the maximum sampling depth in each step

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        Parameters
        ----------
        num_results
            MCMC chain length (default 10000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        max_tree_depth
            Maximum tree depth (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        result object

        result
            Compositional analysis result
        """

        # NUTS transition kernel
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=tf.cast(step_size, tf.float64),
            max_tree_depth=max_tree_depth)
        nuts_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=nuts_kernel,
            bijector=self.constraining_bijectors
        )

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Step size adaptation (Nesterov, 2009)
        nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=num_adapt_steps,
            target_accept_prob=tf.cast(0.75, tf.float64),
            decay_rate=0.75,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        nuts_kernel = tfp.experimental.mcmc.WithReductions(nuts_kernel, pbar)

        # trace function
        def trace_fn(_, pkr):
            return {
                "target_log_prob": pkr.inner_results.inner_results.inner_results.target_log_prob,
                "leapfrogs_taken": pkr.inner_results.inner_results.inner_results.leapfrogs_taken,
                "diverging": pkr.inner_results.inner_results.inner_results.has_divergence,
                "energy": pkr.inner_results.inner_results.inner_results.energy,
                "log_accept_ratio": pkr.inner_results.inner_results.inner_results.log_accept_ratio,
                "step_size": pkr.inner_results.inner_results.inner_results.step_size[0],
                "reach_max_depth": pkr.inner_results.inner_results.inner_results.reach_max_depth,
                "is_accepted": pkr.inner_results.inner_results.inner_results.is_accepted,
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, num_burnin, nuts_kernel, self.init_params, trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin, is_nuts=True)
        pbar.bar.close()

        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {
            "chain_length": num_results,
            "num_burnin": num_burnin,
            "acc_rate": acc_rate,
            "duration": duration,
            "y_hat": y_hat
        }

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def make_result(self, states_burnin, sample_stats, sampling_stats):

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation setup
        # Get names of cell types that are not the reference
        cell_types_nb = self.node_names.copy()
        for ct in self.reference_nodes:
            cell_types_nb.remove(self.node_names[ct])

        cell_types_select = self.node_names

        # Result object generation process. Uses arviz's data structure.
        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}

        if "prediction" in self.param_names:
            posterior_predictive = {"prediction": [params["prediction"]]}
        else:
            posterior_predictive = {}

        observed_data = {"y": self.y}

        dims = {
            "alpha_0": ["covariate", "cell_type_nb"],
            "b_0": ["covariate", "cell_type_nb"],
            "alpha_1": ["covariate", "cell_type_nb"],
            "b_1": ["covariate", "cell_type_nb"],
            "theta": ["x"],
            "alpha": ["cell_type"],
            "bet_0": ["covariate", "cell_type_nb"],
            "bet_1": ["covariate", "cell_type_nb"],
            "beta_select": ["covariate", "cell_type_select"],
            "beta": ["covariate", "cell_type"],
            "concentration": ["sample", "cell_type"],
            "prediction": ["sample", "cell_type"]
        }
        coords = {
            "cell_type": self.cell_types,
            "cell_type_nb": cell_types_nb,
            "cell_type_select": cell_types_select,
            "covariate": self.covariate_names,
            "sample": range(self.y.shape[0])
        }

        model_specs = {
            "reference": self.reference_cell_type,
            "reference_nodes": self.reference_nodes,
            "formula": self.formula,
            "A": self.A,
            "spike_slab": False,
            "lambda_0": self.l_0,
            "lambda_1": self.l_1
        }

        return res.CAResultConverter_tree(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            observed_data=observed_data,
            dims=dims,
            sample_stats=sample_stats,
            coords=coords
        ).to_result_data(sampling_stats=sampling_stats, model_specs=model_specs)