PyAutoLens Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD

|binder|

`Installation Guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautolens.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/3b48dbc1b0ee85e68a24394895702df78e465323?filepath=introduction.ipynb>`_ |
`HowToLens <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_

Welcome to the **PyAutoLens** Workspace. You can get started right away by going to the `autolens workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautolens.readthedocs.io/>`_.

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoLens** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoLens** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoLens**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed with **PyAutoLens**.
- ``output``: Where the **PyAutoLens** analysis and visualization are output.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``overview``: Examples giving an overview of **PyAutoLens**'s core features.
- ``howtolens``: Detailed step-by-step Jupyter notebook tutorials on how to use **PyAutoLens**.
- ``imaging``: Examples for analysing and simulating CCD imaging data.
- ``interferometer``: Examples for analysing and simulating interferometer.
- ``database``: Examples for using database tools which load libraries of model-fits to large datasets.
- ``plot``: An API reference guide for **PyAutoLens**'s plotting tools.
- ``misc``: Miscelaneous scripts for specific lens analysis.

In the ``imaging`` and ``interferometer`` folders you'll find the following packages:

- ``modeling``: Examples of how to fit a lens model to data via a non-linear search.
- ``simulators``: Scripts for simulating realistic imaging and interferometer data of strong lenses.
- ``preprocess``: Tools to preprocess ``data`` before an analysis (e.g. convert units, create masks).

The ``advanced`` sections are for veteran users and contain:

- ``pipelines``: Example pipelines for modeling strong lenses using non-linear search chaining.
- ``SLaM``: The Source, Light and Mass (SLaM) lens modeling pipelines.

Getting Started
---------------

We recommend new users begin with the example notebooks / scripts in the *overview* folder and the **HowToLens**
tutorials.

Workspace Version
-----------------

This version of the workspace are built and tested for using **PyAutoLens v1.12.0**.

HowToLens
---------

Included with **PyAutoLens** is the ``HowToLens`` lecture series, which provides an introduction to strong gravitational
lens modeling with **PyAutoLens**. It can be found in the workspace & consists of 5 chapters:

- ``Introduction``: An introduction to strong gravitational lensing & **PyAutoLens**.
- ``Lens Modeling``: How to model strong lenses, including a primer on Bayesian non-linear analysis.
- ``Pipelines``: How to build model-fitting pipelines & tailor them to your own science case.
- ``Inversions``: How to perform pixelized reconstructions of the source-galaxy.
- ``Hyper-Mode``: How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.


Contribution
------------
To make changes in the tutorial notebooks, please make changes in the the corresponding python files(.py) present in the
``scripts`` folder of each chapter. Please note that  marker ``# %%`` alternates between code cells and markdown cells.


Support
-------

Support for installation issues, help with lens modeling and using **PyAutoLens** is available by
`raising an issue on the autolens_workspace GitHub page <https://github.com/Jammy2211/autolens_workspace/issues>`_. or
joining the **PyAutoLens** `Slack channel <https://pyautolens.slack.com/>`_, where we also provide the latest updates on
**PyAutoLens**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.