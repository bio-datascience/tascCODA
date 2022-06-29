About tascCODA
==============

When analyzing population data from high-throughput sequencing (HTS) experiments, e.g. single-cell RNA-seq or 16S rRNA sequencing,
it is often of interest to see how the cell or microbial population changes in response to some sample-specific covariates.
These covariates can be indicators of certain diseases or treatments, host-specific measurements like BMI or age, or environmental factors such as temperature.

Compositional data analysis in HTS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When doing differential population analysis, one property of cell- or microbial abundance data is often overlooked. Since all
HTS platforms are limited in their throughput, the number of counts in a sample (i.e. the library size) is
predetermined. Thus, HTS populations are compositional. They can only be determined up to a multiplicative factor, inducing a negative
correlative bias between the cell types. Following
`Aitchison (Journal of the Royal Statistical Society, 1982) <https://www.jstor.org/stable/2345821?seq=1>`_,
compositional data thus has to be interpreted in terms of ratios, e.g. with respect to a reference factor.

Features of tascCODA
^^^^^^^^^^^^^^^^^^^^

The tascCODA model (`Ostner et al. (2021) <https://www.frontiersin.org/articles/10.3389/fgene.2021.766405/full>`_)
is a model that was specifically designed to perform tree-aggregated compositional data analysis on high-throughput sequencing data.
Apart from the compositionality of the data, there are some other challenges that all HTS datasets have in common.
These include a rather low number of samples and an excess of zero-entries.

Since the number of features (cell types/OTUs/ASVs/...) and therefore the number of possible covariate-feature interactions is often very large,
it is important to not only infer the effect of each covariate on each feature, but also to determine whether these effects significantly impact the population.

Furthermore, most HTS datasets come with a hierarchical ordering that clusters biologically or genetically similar features together.
tascCODA uses these clusterings, for example phylogenetic trees or cell lineage hierarchies, to infer common compositional shifts on subsets of the features
whenever possible.
This leads to a better understanding of how the covariates affect the data on all levels of the hierarchy without the need to run the analysis on fixed-level aggregations.

tascCODA is a tree-aggregated extension of the scCODA model, which uses Bayesian modeling and its possibility to include prior beliefs to obtain accurate results even in a low-sample setting,
while performing automatic model selection via a spike-and-slab prior setup.

Just as in scCODA, tascCODA allows the user to select any reference cell type in order to see the effects
of biological factors from different perspectives.
Furthermore, the level and strength of tree-aggregation can be adjusted to favor higher- or lower-level aggregations.

For more detailed information on the tascCODA model, see
`Ostner et al. (2021) <https://www.frontiersin.org/articles/10.3389/fgene.2021.766405/full>`_.

