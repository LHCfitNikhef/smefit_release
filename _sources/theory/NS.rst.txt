Nested Sampling
===============


|SMEFiT| implements one fitting strategies using the MultiNest and :doc:`Nested Sampling<NS>`
algorithm :cite:`Feroz:2007kg,Feroz:2013hea`, a robust sampling procedure is based on Bayesian inference.


Formalism
~~~~~~~~~

The starting point of Nested Sampling (|NS|) is Bayes' theorem, which allows us to evaluate the
probability distribution of a set of parameters :math:`\vec{c}`
associated to a model :math:`\mathcal{M}(\vec{c})`
given a set of experimental measurements :math:`\mathcal{D}` ,

.. math::

   P\left(\vec{c}| \mathcal{D},\mathcal{M} \right) = \frac{P\left(\mathcal{D}|\mathcal{M},\vec{c}  \right) P\left( \vec{c}|\mathcal{M}  \right) }{P(\mathcal{D}|\mathcal{M})} \, .


Here :math:`P\left(\vec{c}| \mathcal{D},\mathcal{M} \right)`  represents the posterior
probability of the model parameters given the assumed model and the observed
experimental data,
:math:`P\left(\mathcal{D}|\mathcal{M},\vec{c}\right) = \mathcal{L}\left(\vec{c} \right)`  is the likelihood (conditional
probability) of the experimental measurements
given the model and a specific choice of parameters,
and :math:`P\left( \vec{c}|\mathcal{M}  \right) = \pi \left(  \vec{c} \right)`
is the prior distribution for the model parameters.

The denominator in the equation above, :math:`P(\mathcal{D}|\mathcal{M}) = \mathcal{Z}` ,
is known as the Bayesian evidence and ensures the normalization of the posterior distribution,

.. math::

   \mathcal{Z} = \int \mathcal{L}\left(  \vec{c} \right)\pi \left(  \vec{c} \right) d \vec{c} \, ,


where the integration is carried out over the domain of the model parameters :math:`\vec{c}` .

The key ingredient of |NS| utilise the ideas underlying
Bayesian inference to map the :math:`n` -dimensional integral over the prior density
in model parameter space :math:`\pi(\vec{c} )d\vec{c}` ,
where :math:`n`  represents the dimensionality of :math:`\vec{c}` , into a one-dimensional function
of the form

.. math::

   X(\lambda) = \int_{\{ \vec{c} : \mathcal{L}\left(\vec{c} \right) > \lambda \}}\pi(\vec{c} ) d\vec{c} \,.


In this expression, the prior mass :math:`X(\lambda)`  corresponds to the
(normalized) volume
of the prior density :math:`\pi(\vec{c} )d\vec{c}`  associated with values
of the model parameters that lead to a likelihood :math:`\mathcal{L}\left(\vec{c}\right) `  greater
than the parameter :math:`\lambda` .
Note that by construction, the prior mass :math:`X`  decreases monotonically
from the limiting value :math:`X=1`  to :math:`X=0`  as :math:`\lambda`  is increased.
The integration of :math:`X(\lambda)`  extends over the regions in the model parameter space contained
within the fixed-likelihood contour defined by the condition :math:`\mathcal{L}\left(\vec{c}\right) =\lambda`.

This property allows the evidence to be expressed as,

.. math::
   \mathcal{Z} = \int_0^1  \mathcal{L}\left( X\right) dX \, ,

where :math:`\mathcal{L}\left( X\right)`  is defined as the inverse function of :math:`X(\lambda)` , which
always exists provided the likelihood is a continuous and smooth function
of the model parameters.
Therefore, the transformation from :math:`\vec{c}`
to X  achieves a mapping of the prior distribution into infinitesimal
elements, sorted by their associated likelihood :math:`\mathcal{L}(\vec{c})` .

The next step of the NS algorithm is to define a decreasing sequence of values in the prior
volume, that is now parameterized by the prior mass :math:`X` .
In other words, one slices the prior volume into a large number of
small regions

.. math::
   1 = X_0 > X_1 > \ldots X_{\infty} = 0 \, ,

and then evaluates the likelihood at each of these values, :math:`\mathcal{L}=\mathcal{L}(X_i)` .

This way, all of the :math:`\mathcal{L}_i`  values
can be summed in order to  evaluate the integral
for the Bayesian evidence.
Since in general the likelihood :math:`\mathcal{L}({\boldsymbol c})`  exhibits a complex dependence
on the model parameters :math:`\vec{c}` , the summation
above must be evaluated
numerically using {\it e.g.} Monte Carlo integration methods.

In practice, one draws :math:`N_{\rm live}`  points from the parameter prior
volume :math:`\pi\left(\vec{c} \right)` , known as {\it live points}, and orders
the likelihood values from smallest to largest, including the
starting value of the prior mass at :math:`X_0=1`. As samples are drawn from the prior volume,
the live point with the lowest likelihood :math:`\mathcal{L}_i`
is removed from the set and replaced by another live point drawn from the same prior
distribution but now under the constraint that its likelihood is larger than
:math:`\mathcal{L}_i`. This sampling process is repeated until the entire hyper-volume
:math:`\pi \left( \vec{c} \right)`  of the prior parameter space has been covered, with ellipsoids of constrained likelihood being assigned to the live-points as the prior volume is scanned. While the end result of the NS procedure is the estimation of the  Bayesian evidence :math:`\mathcal{Z}` , as a byproduct one also obtains  a sampling of the posterior distribution associated to the EFT coefficients expressed as

.. math::

   \{ \vec{c}^{(k)} \}\, ,\qquad  k=1,\dots,N_{\rm spl}\, ,

with :math:`N_{\rm spl}`  indicating the number
of samples drawn by the final NS iteration.

One can then compute expectation values and variances of the model
parameters by evaluating the MC sum over the these posterior samples together
with their associated weights, in the same
way as averages are done
over the :math:`N_{\rm rep}` replicas in the MCfit method.

Prior volume
~~~~~~~~~~~~

An important input for NS is the choice of prior volume :math:`\pi \left( \vec{c} \right)`
in the model parameter space.
In this analysis, we adopt flat priors
defined by ranges in parameter space for the coefficients :math:`\vec{c}` .
A suitable choice of prior volume where the sampling takes place is important
to speed up the NS algorithm: a range too wide will make the optimization less
efficient, while a range too narrow might bias the results by cutting
specific regions of the parameter space that are relevant.
Furthermore, using a common range for all parameters should be avoided,
since the range of intrinsic variation will be rather different for each
of the EFT coefficients.

Taking these considerations into account, we adopt here the following strategy.
First, a single model parameter :math:`c_i`  is allowed to vary
while all others are set to their SM value, :math:`c_j=0`  for :math:`j\ne i` .
The :math:`\chi^2 \left( c_i \right)`  is then scanned in this
direction to determine the values :math:`c_i^{\rm (min)}`  and  :math:`c_i^{\rm (max)}`
satisfying the condition :math:`\chi^2/n_{\rm dat}=4` .
We then repeat this procedure
for all parameters and end up with a hyper-volume
defined by pairs of values :math:`\left( c_i^{\rm (min)},c_i^{\rm (max)} \right)`  with
:math:`i=1,\ldots, n_{\rm op}`  which defines our initial prior volume.

At this point, one performs an initial exploratory NS global analysis using this
volume to study the posterior probability distribution
for each EFT coefficient.
Our final analysis is then obtained by manually adjusting the
initial sampling ranges until the full posterior distributions are captured for the chosen
prior volume.
For parameters that are essentially unconstrained in the global fit,
such as the four-heavy operators in the case of linear EFT calculations,
a hard boundary of :math:`\left( -50 , 50 \right)`  (for :math:`\Lambda=1`  TeV) is imposed.

Performance
~~~~~~~~~~~


To increase the efficiency of the posterior probability estimation by NS, we enable the
*constant efficiency mode* in MultiNest, which adjusts the
total volume of ellipsoids spanning the live points so that the sampling
efficiency is close to its associated hyperparameter set by the user.
With 24 cpu cores,
we are able to achieve an accurate posterior for the linear EFT fits
in :math:`\sim 30`  minutes using 500 live points, a target efficiency of 0.05, and
an evidence tolerance of 0.5, which results in :math:`N_{\rm spl}\simeq 5000`  posterior samples.
To ensure the stability of our final results, we chose
1000 live points and a target efficiency of
0.005, which yields :math:`\simeq 1.5\times 10^4`  samples for the
linear analysis and :math:`\simeq 10^4`  samples for an analysis that includes also the quadratic EFT
corrections.
With these settings, our final global analyses containing the simultaneous
determination of :math:`n_{\rm op}\simeq 36`  coefficients
take :math:`\sim 3.5`  hours running in 24 cpu cores, with a similar performance for
linear and quadratic EFT fits.
