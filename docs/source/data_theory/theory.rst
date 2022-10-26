Theory tables
=============
Each experimental dataset is associate with a corresponding theory table which has to be provided by the user.
Theory tables are json files containing the following information:

* best standard model predictions for each datapoint: provided as a list ``[best_sm_prediction_1, ... ,best_sm_prediction_N]`` with ``N`` being the number of datapoints

* theory covariance matrix for the specific dataset considered: provided as a ``N x N`` matrix like ``[[th_cov_11, ... ,th_cov_1N], ... , [th_cov_N1, ... ,th_cov_NN]]`` with ``N`` being the number of data points

* LO and NLO predictions with linear and quadratic SMEFT corrections: provided as two independent dictionaries ``LO: {}`` and ``NLO: {}`` each containing
  
  * SM predictions obtained at that specific order ``SM: [sm_prediction_1, ... , sm_prediction_N]``
  * linear terms for each operator involved in the computation ``Opi: [linear_term_Opi_1, ..., linear_term_Opi_N]``
  * quadratic terms for each couple of operators (when present) ``Opi*Opj: [quad_term_Opi*Opj_1, ..., quad_term_Opi*Opj_N]``
 
The EFT corrections should always be provided in the Warsaw basis. In order to produce a fit with a different basis,
the corresponding rotation matrix has to be provided externally, see :ref:`here<rotation>` for more details.

The above information are used to build the theory predictions for the different observables 
entering the :math:`\chi^2`. As default theory predictions are expressed as

.. math::

  \sigma=\sigma_{\rm SM}^{\rm best} + \sum_i^{N_{d6}}{\kappa_i}^{\rm LO/NLO} \frac{c_i}{\Lambda^2} + \sum_{i,j}^{N_{d6}} \widetilde{\kappa}_{ij}^{\rm LO/NLO} \frac{c_ic_j}{\Lambda^4}  \, ,



The user can also choose to use the relation

.. math::

   \sigma=\sigma_{\rm SM}^{\rm best}\left(1 + \sum_i^{N_{d6}}\frac{{\kappa_i}^{\rm LO/NLO}}{\sigma_{\rm SM}^{\rm LO/NLO}} \frac{c_i}{\Lambda^2} + \sum_{i,j}^{N_{d6}}  \frac{\widetilde{\kappa}_{ij}^{\rm LO/NLO}}{\sigma_{\rm SM}^{\rm LO/NLO}} \frac{c_ic_j}{\Lambda^4}\right)  \, ,


instead by setting in the runcard ``use_multiplicative_prescription: True``.
The reason to take the ratios :math:`\frac{{\kappa_i}^{\rm LO/NLO}}{\sigma_{\rm SM}^{\rm LO/NLO}}` and :math:`\frac{\widetilde{\kappa}_{ij}^{\rm LO/NLO}}{\sigma_{\rm SM}^{\rm LO/NLO}}`
is to reduce the dependence on the specific set of parton distribution functions used to computed the 
LO and NLO predictions.
