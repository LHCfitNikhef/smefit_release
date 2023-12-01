Analytic solution
=================

In case only linear |EFT| corrections are needed or available, |SMEFiT| 
allows you to obtain the analytic solution of the linear problem.

The central value of the Wilson coefficient is given by solving the equation :math:`\partial \chi^2 / \partial c = 0`:

.. math::
   c_i = \left (  \kappa_{i,k} \text{cov}_{k,l}^{-1}  \kappa_{j,l} \right )^{-1} \kappa_{j,m} \text{cov}_{m,n}^{-1} \left ( \sigma_{n}^{\text{(exp)}} -  \sigma_{n}^{\text{(sm)}} \right ) \quad i,j=\{1,\dots N_{op}\}, \quad m,n,k,l=\{1,\dots N_{dat}\}

and its covariance matrix is given by the Fisher information: 

.. math::
   X_{ij} = \kappa_{i,k} \text{Cov}_{k,l}^{-1} \kappa_{j,l} \quad i,j=\{1,\dots N_{op}\}


Starting from the central values and the covariance one can then draw the required number of samples from a muli Gaussian distribution :math:`\mathcal{N}(c, X)`. 

This analytic method is really efficient, however it does not work in the presence of flat 
directions, which can result in a :math:`X_{ij}` matrix not semipositive definite.

Also quadratic relastion between Wilson coefficient, as well as quadratic |EFT| corrections are not supported.