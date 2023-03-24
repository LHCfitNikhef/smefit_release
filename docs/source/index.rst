
.. image:: ../_assets/logo.png
  :width: 400
  :align: center
  :alt: SMEFIT logo

Project description
~~~~~~~~~~~~~~~~~~~

|SMEFiT| is a Python package for global analyses of particle physics data in the framework of the Standard Model Effective Field Theory (|SMEFT|).
The |SMEFT| represents a powerful model-independent framework to constrain, identify,
and parametrize potential deviations with respect to the predictions of the Standard Model (SM).
A particularly attractive feature of the |SMEFT| is its capability to systematically correlate deviations
from the SM between different processes. The full exploitation of the |SMEFT| potential for indirect
New Physics searches from precision measurements requires combining the information provided by the broadest possible dataset,
namely carrying out extensive global analysis which is the main purpose of |SMEFiT|.


The SMEFiT framework has been used in the following **scientific publications**:

- *A Monte Carlo global analysis of the Standard Model Effective Field Theory: the top quark sector*, N. P. Hartland, F. Maltoni, E. R. Nocera, J. Rojo, E. Slade, E. Vryonidou, C. Zhang :cite:`Hartland:2019bjb`.
- *Constraining the SMEFT with Bayesian reweighting*, S. van Beek, E. R. Nocera, J. Rojo, and E. Slade :cite:`vanBeek:2019evb`.
- *SMEFT analysis of vector boson scattering and diboson data from the LHC Run II* , J. Ethier, R. Gomez-Ambrosio, G. Magni, J. Rojo :cite:`Ethier_2021`.
- *Combined SMEFT interpretation of Higgs, diboson, and top quark data from the LHC*, J. Ethier, G.Magni, F. Maltoni, L. Mantani, E. R. Nocera, J. Rojo, E. Slade, E. Vryonidou, C. Zhang :cite:`ethier2021combined`

Results from these publications, including driver and analysis scripts, are available in the *Previous studies* section.


When using the code please cite:

- *SMEFiT: a flexible toolbox for global interpretations of particle physics data with effective field theories*, T. Giani, G. Magni and J. Rojo, :cite:`Giani:2023gfq`

.. toctree::
    :caption: Theory:
    :maxdepth: 1
    :hidden:

    theory/SMEFT.rst
    theory/general
    theory/NS
    theory/MCFit


.. toctree::
    :maxdepth: 2
    :caption: Data and theory tables:
    :hidden:

    data_theory/data.md
    data_theory/theory
    data_theory/covariance
    data_theory/rotation.md

.. toctree::
    :maxdepth: 3
    :caption: Fitting code:
    :hidden:

    fitting_code/code_struct.md
    fitting_code/running.md

.. toctree::
    :maxdepth: 2
    :caption: Reports:
    :hidden:

    report/tools
    report/running.md
    report/links


.. toctree::
    :maxdepth: 1
    :caption: Previous studies:
    :hidden:

    previous_releases/smefit_rw
    previous_releases/smefit_top
    previous_releases/smefit_vbs
    previous_releases/smefit20

.. toctree::
    :maxdepth: 1
    :caption: Collaboration:
    :hidden:

    people/people.rst

.. toctree::
    :maxdepth: 1
    :caption: References & API:
    :hidden:

    API <modules/smefit/smefit>
    zzz-refs


Indices and tables
==================

* :ref:`genindex`
* :doc:`API </modules/smefit/smefit>`
* :ref:`search`
