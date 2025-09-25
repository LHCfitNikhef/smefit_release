
.. image:: ../_assets/logo.png
  :width: 400
  :align: center
  :alt: SMEFIT logo

Project description
===================

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
- *Combined SMEFT interpretation of Higgs, diboson, and top quark data from the LHC*, J. Ethier, G.Magni, F. Maltoni, L. Mantani, E. R. Nocera, J. Rojo, E. Slade, E. Vryonidou, C. Zhang :cite:`ethier2021combined`.
- *The automation of SMEFT-assisted constraints on UV-complete models*, J. ter Hoeve, G. Magni, J. Rojo, A. N. Rossia, E. Vryonidou :cite:`terHoeve:2023pvs`.
- *Mapping the SMEFT at High-Energy Colliders: from LEP and the (HL-)LHC to the FCC-ee*, E.Celada, T. Giani, J. ter Hoeve, L. Mantani, J. Rojo, A. N. Rossia, M. O. A. Thomas, E. Vryonidou :cite:`Celada:2024mcf`.
- *Connecting Scales: RGE Effects in the SMEFT at the LHC and Future Colliders*, J. ter Hoeve, L. Mantani, A. N. Rossia, J. Rojo, E. Vryonidou :cite:`terHoeve:2025gey`.
- *The Higgs trilinear coupling in the SMEFT at the HL-LHC and the FCC-ee*, J. ter Hoeve, L. Mantani, A. N. Rossia, J. Rojo, E. Vryonidou :cite:`Hoeve:2025yup`.
Results from these publications, including driver and analysis scripts, are available in the *Previous studies* section.

When using the code please cite:

- *SMEFiT: a flexible toolbox for global interpretations of particle physics data with effective field theories*, T. Giani, G. Magni and J. Rojo, :cite:`Giani:2023gfq`

An introductory overview of SMEFiT was recently presented at the following workshop:

.. raw:: html

    <video
      controls
      style="width:100%; max-width:700px; height:auto; display:block;">
      <source src="https://indico.cern.ch/event/1587395/attachments/3142763/5578327/GMT20250924-120121_Recording_1920x1200.mp4" type="video/mp4">
    </video>

Installation
------------

To install the smefit release on PYPI you can do:

.. code-block:: bash

    pip install smefit

Installation for Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in developing smefit or having the latest smefit code not yet released, you should clone the smefit repository and then install in editable mode:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/LHCfitNikhef/smefit_release.git
       cd smefit_release

2. Install in editable mode:

   .. code-block:: bash

       pip install -e .

To use a Conda environment (e.g., Python 3.12):

.. code-block:: bash

    conda create -n smefit-dev python=3.12
    conda activate smefit-dev
    pip install -e .


.. toctree::
    :caption: Theory:
    :maxdepth: 1
    :hidden:

    theory/SMEFT.rst
    theory/general
    theory/NS
    theory/Analytic
    theory/MCFit


.. toctree::
    :maxdepth: 2
    :caption: Data and theory tables:
    :hidden:

    data_theory/data.md
    data_theory/theory
    data_theory/covariance
    data_theory/rotation.md
    data_theory/projections

.. toctree::
    :maxdepth: 3
    :caption: Fitting code:
    :hidden:

    fitting_code/code_struct.md
    fitting_code/running.md
    fitting_code/tutorial

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
    previous_releases/smefit_uv
    previous_releases/smefit30

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
