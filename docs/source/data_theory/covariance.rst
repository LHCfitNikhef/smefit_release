Construction of the fit covariance matrix
=========================================

In the following we provide an explicit example of how the covariance matrix is built, 
given two datasets having both uncorrelated and correlated systematics.
We consider 2 datasets having respectively 2 datapoints with 3 systematic uncertainties, and  
3 datapoints with 2 systematic uncertainties.
The total statistic uncertainties for the two datasets are denoted as

.. math::

   \sigma_i\,, \,\,\, i = 1,2 \,\,\,\,\,\,\,\, \text{and} \,\,\,\,\,\,\,\,  \bar{\sigma}_j \,,\,\,\, j = 1,2,3
  

while the systematic uncertainties as

.. math::

   \sigma^{\text{sys},\theta}_i\,, \,\,\,\, i = 1,2\,, \,\,\,\, \theta = 1,2,3\,,
   \,\,\,\,\,\,\,\, \text{and} \,\,\,\,\,\,\,\,
   \bar{\sigma}^{\text{sys},\theta}_i\,, \,\,\,\,\,\,\,\, i = 1,2,3\,, \,\,\,\, \theta = 1,2\,,

with the upper and lower indices labelling the systematic and the datapoint respectively.
We further assume that 

* the systematic :math:`\sigma^{\text{sys},1}_i\,,\,\,\, \sigma^{\text{sys},2}_i` of the first dataset  
  are correlated within the dataset, and therefore named ``CORR`` according to our convention, 
  while :math:`\sigma^{\text{sys},3}_i` is named as ``SPECIAL``

* regarding the second dataset, the systematic :math:`\bar{\sigma}^{\text{sys},1}_i` is taken as uncorrelated,
  and therefore denoted as ``UNCORR``, while :math:`\bar{\sigma}^{\text{sys},2}_i` is named ``SPECIAL``, and 
  therefore will be correlated with the one of the other dataset having the same name


Considering a single dataset, the covariance matrix is given by

.. math::

    \text{cov}_{ij} = \sigma_i\,\sigma_j \, \delta_{ij} + \sum_{\theta}\sigma^{\text{sys},\theta}_i \sigma^{\text{sys},\theta}_j 

where the sum on :math:`\theta` runs on the correlated systematics between the point :math:`i\,,\,\,j`. 
When considering the two datasets together, additional off diagonal contributions should be added to account for 
the correlated systematics denoted as ``SPECIAL``.
Specifically we have, from the statistical, ``CORR`` and ``UNCORR`` uncertainties

.. math::
    
    \begin{pmatrix}
    \sigma_1^2 & 0 & 0 & 0 & 0 \\
    0 & \sigma_2^2 & 0 & 0 & 0 \\
    0 & 0 & \bar{\sigma}_1^2 & 0 & 0 \\
    0 & 0 & 0 & \bar{\sigma}_2^2 & 0 \\
    0 & 0 & 0 & 0 & \bar{\sigma}_3^2 
    \end{pmatrix}
    +
    \begin{pmatrix}
    {\left(\sigma^{\text{sys},1}_1\right)}^2 + {\left(\sigma^{\text{sys},2}_1\right)}^2   & \sigma^{\text{sys},1}_1 \sigma^{\text{sys},1}_2 + \sigma^{\text{sys},2}_1 \sigma^{\text{sys},2}_2 & 0 & 0 & 0 \\
    \sigma^{\text{sys},2}_1 \sigma^{\text{sys},1}_1 + \sigma^{\text{sys},2}_2 \sigma^{\text{sys},2}_1  & {\left(\sigma^{\text{sys},1}_2\right)}^2 + {\left(\sigma^{\text{sys},2}_2\right)}^2 & 0 & 0 & 0 \\
    0 & 0 & {\left(\bar{\sigma}^{\text{sys},1}_1\right)}^2 & 0 & 0 \\
    0 & 0 & 0 & {\left(\bar{\sigma}^{\text{sys},1}_2\right)}^2 & 0 \\
    0 & 0 & 0 & 0 & {\left(\bar{\sigma}^{\text{sys},1}_3\right)}^2
    \end{pmatrix}

while from the cross correlated systematic

.. math::

    \begin{pmatrix}
    \left(\sigma_1^{\text{sys},3}\right)^2 & \sigma_1^{\text{sys},3}\sigma_2^{\text{sys},3} & \sigma_1^{\text{sys},3}\bar{\sigma}_1^{\text{sys},2} & \sigma_1^{\text{sys},3}\bar{\sigma}_2^{\text{sys},2} & \sigma_1^{\text{sys},3}\bar{\sigma}_3^{\text{sys},2} \\
    \sigma_2^{\text{sys},3}\sigma_1^{\text{sys},3} & \left(\sigma_2^{\text{sys},3}\right)^2 & \sigma_2^{\text{sys},3}\bar{\sigma}_1^{\text{sys},2} & \sigma_2^{\text{sys},3}\bar{\sigma}_2^{\text{sys},2} & \sigma_2^{\text{sys},3}\bar{\sigma}_3^{\text{sys},2} \\    
    \bar{\sigma}_1^{\text{sys},2}\sigma_1^{\text{sys},3} & \bar{\sigma}_1^{\text{sys},2}\sigma_2^{\text{sys},3} & \left(\bar{\sigma}_1^{\text{sys},2}\right)^2 & \bar{\sigma}_1^{\text{sys},2}\bar{\sigma}_2^{\text{sys},2} & \bar{\sigma}_1^{\text{sys},2}\bar{\sigma}_3^{\text{sys},2} \\
    \bar{\sigma}_2^{\text{sys},2}\sigma_1^{\text{sys},3} & \bar{\sigma}_2^{\text{sys},2}\sigma_2^{\text{sys},3} & \bar{\sigma}_2^{\text{sys},2}\bar{\sigma}_1^{\text{sys},2} & \left(\bar{\sigma}_2^{\text{sys},2}\right)^2 & \bar{\sigma}_2^{\text{sys},2}\bar{\sigma}_3^{\text{sys},2} \\
    \bar{\sigma}_3^{\text{sys},2}\sigma_1^{\text{sys},3} & \bar{\sigma}_3^{\text{sys},2}\sigma_2^{\text{sys},3} & \bar{\sigma}_3^{\text{sys},2}\bar{\sigma}_1^{\text{sys},2} & \bar{\sigma}_3^{\text{sys},2}\bar{\sigma}_2^{\text{sys},2} & \left(\bar{\sigma}_3^{\text{sys},2}\right)^2
    \end{pmatrix}

