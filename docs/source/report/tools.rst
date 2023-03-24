Report functions
================

In this section we describe some of the analysis tools that are provided in the report.
Alognside :math:`\chi^2` tables dataset per dataset, correlation plots between the coefficients
and various histograms to represent the posterior distribution, we include two other main statistical
estimators.

PCA
---

The Principal Components Analysis (|PCA|) is the singular values decomposition of the matrix defined as:

.. math ::
    X_{ij} = \kappa_{i,k} \text{Cov}_{k,l}^{-1} \kappa_{j,l} \quad i,j=\{1,\dots N_{op}\}, \quad l,l=\{1,\dots N_{dat}\}

where :math:`\kappa_{i,k}` are the linear |EFT| contibution for each operator and datapoint,
:math:`\text{Cov}_{k,l}^{-1}` is the inverse of the total covariance matrix.

Then :math:`X_{ij}` is decomposed as:

.. math ::
    X_{ij}^{(center)} &=  X_{ij} - \frac{1}{N_{op}} \sum_{k=0}^{N_{op}} X_{kj} \\
    X_{ij}^{(center)} &= U_{i,k} S_{k} V^{T}_{k,j}

where the vector :math:`S_{k}` is the vector of the eigenvalues squared of :math:`X_{ij}^{(center)}`.

.. math ::
    S_{k} &= \sqrt{D_{k}^2} \\
    X_{ij}^{(center)} &= N_{i,k} D_{k} N_{kj}^T


:math:`V_{i,k}` is the unitary matrix that relates the original basis to the principal components
one. Unlike the eigenvalues, :math:`S_{k}` are positive definite,
and therefore this basis highlighs the directions giving higher or lower weights to the :math:`\chi^2`,
and it can be used to determine possible flat directions.
Note that in order to compute a non biased PCA you need to center the data, see for instance
`this example <https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca>`_.


Fisher information
------------------

The Fisher information matrix is defined as  the expectation values of:

.. math::
    I_{ij} = - \mathbf{E} \left [ \frac{\partial^2 \ln \mathcal{L}}{ \partial c_{i} \partial c_{j} } \right ]

with :math:`\mathcal{L}` is the likelihood function.
This quantity is related through the Cramer Rao bound to the covarince matrix of the Wilson
coefficients :math:`C_{ij}`:

.. math ::
    C_{ij} \ge I^{-1}_{ij}

thus its diagonal entries provide an esitmate of the best achievable uncertainty on each coefficient given the
experimental and theoretical inputs. Note that this inequality is between two matrices,
meaning that :math:`(C - I^{(-1)})_{ij}` is a positive define matrix.
The higher the information value the smaller will be the best error bound
of the coefficient.

At linear level, in case of a multi Gaussian likelihood, we find :math:`I_{ij}=X_{ij}`,
while when quadratics corrections are included this will
depends explicily on the best fitted Wilson coefficients.

.. math ::
    I_{ij} = - \frac{1}{2} \mathbf{E}  \left [ \frac{\partial^2 \sigma_{k}}{\partial c_{i} \partial c_{j}}  \text{Cov}_{k,l}^{-1} \Delta_{l} \right.
        + \frac{\partial \sigma_{k}}{\partial c_{i} } \text{Cov}_{k,l}^{-1} \frac{\partial \Delta_{l}}{\partial c_{j}}
        + \frac{\partial \Delta_{k}}{\partial c_{j}}  \text{Cov}_{k,l}^{-1} \frac{\partial \sigma_{l}}{\partial c_{i}}
        \left. + \Delta_{k}  \text{Cov}_{k,l}^{-1}  \frac{\partial^2 \sigma_{l}}{\partial c_{i} \partial c_{j}} \right]

where we have:

.. math ::
    \sigma_{k} &= \sigma_{\text{SM}, k} + \sum_i^{N_{op}} c_i \kappa_{i,k} + \sum_{i, j\ge i}^{N_{op}} c_i c_{j} \tilde{\kappa}_{ij,k} \\
    \Delta_{k} &= (\sigma_{\text{exp}, k} - \sigma_k) \\
    \frac{\partial^2 \sigma_{k}}{\partial c_{i} \partial c_{j}} &= ( \epsilon_{ij} + 2 \delta_{ij} ) \tilde{\kappa}_{ij,k} \\
    \frac{\partial \Delta_{k}}{\partial c_{i}} &= - \frac{\partial \sigma_{k}}{\partial c_{i}} = - \kappa_{i,k} - \sum_{n \neq i }^{N_{op}} c_{n} \tilde{\kappa}_{in,k} - 2 c_i \tilde{\kappa}_{ii,k}

Expanding all the definitions a looking at the diagonal term we have:

    .. math ::
        I_{ii} = & - 2 ( \tilde{\kappa}_{ii,k} \text{Cov}_{k,l}^{-1}  \mathbf{E} \left[ \Delta_l \right]
                - \mathbf{E} \left[ \Delta_k \right] \text{Cov}_{k,l}^{-1} \tilde{\kappa}_{ii,l} ) \\
                & + \kappa_{i,k} \text{Cov}_{k,l}^{-1} A_{i,l} + A_{i,k}  \text{Cov}_{k,l}^{-1} \kappa_{i,l} \\
                & + 2 ( \mathbf{E} \left[ c_i \right] \kappa_{i,k} \text{Cov}_{k,l}^{-1} \tilde{\kappa}_{ii,l}
                + \tilde{\kappa}_{ii,k} \text{Cov}_{k,l}^{-1} \mathbf{E} \left[ c_i \right] \kappa_{i,l} ) \\
                & + 2 ( \tilde{\kappa}_{ii,k} \text{Cov}_{k,l}^{-1} B_{i,l} + B_{i,k} \text{Cov}_{k,l}^{-1} \tilde{\kappa}_{ii,k} ) \\
                & + 4 \tilde{\kappa}_{ii,k} \text{Cov}_{k,l}^{-1} \tilde{\kappa}_{ii,k} \mathbf{E} \left[ c^2_i \right]
                + D_{i,kl} \text{Cov}_{k,l}^{-1}

where:

    .. math ::
        A_{i,k} &= \mathbf{E} \left[ \sum_{n \neq i }^{N_{op}} c_n \tilde{\kappa}_{in,k} \right] \\
        B_{i,k} &= \mathbf{E} \left[ \sum_{n \neq i }^{N_{op}} c_i c_n \tilde{\kappa}_{in,k} \right] \\
        D_{i,kl} &= \mathbf{E} \left[ \sum_{n \neq i }^{N_{op}} \sum_{m \neq i }^{N_{op}} c_m c_n \tilde{\kappa}_{in,k} \tilde{\kappa}_{im,l} \right] \\


Inside the code :math:`I_{i,i}` is computed dataset per dataset and normalized either by coefficient or by dataset.
In the latter case this indicates how much each dataset contributes in the determination of a given coefficient,
in the former case it rappresents how much a given dataset is sensitive to each operators.

Finally the Fisher information matrix can also be understood as a metric in model space.
If one has two sets of coefficients :math:`\boldsymbol{c}_a` and :math:`\boldsymbol{c}_b`,
corresponding to two different points in the EFT parameter space,
then the local distance between them is defined as:

.. math ::
    d_{\rm loc}(\boldsymbol{c}_a,\boldsymbol{c}_b) = \left [ \sum_{ij} (\boldsymbol{c}_a-\boldsymbol{c}_b)_i I_{ij}(\boldsymbol{c}_a) (\boldsymbol{c}_a-\boldsymbol{c}_b)_j \right ]^{1/2}

a feature which provides a robust method to quantify how (di)similar
are two points in this model space.


Coefficient bounds
------------------

 * Describe how the CL tables are produced
 * Describe the 2d plots
