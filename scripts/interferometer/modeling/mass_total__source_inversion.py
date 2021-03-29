"""
Modeling: Mass Total + Source Inversion
=======================================

In this script, we fit `Interferometer` data with a strong lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `VoronoiMagnification` `Pixelization` and `Constant`
   regularization.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `mass_sie__source_sersic` from .fits files , which we will fit 
with the lens model.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
)

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

"""
__Masking__

The perform an interferometer model-fit we require two masks: 

 1) A ‘real_space_mask’ which defines the grid the image of the lensed source galaxy is evaluated using.
 2) A ‘visibilities_mask’ defining which visibilities are omitted from the chi-squared evaluation (in this case, none).
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

"""
We now create the `MaskedInterferometer` object which is used to fit the lens model.

This includes a `SettingsMaskedInterferometer`, which includes the method used to Fourier transform the real-space 
image of the strong lens to the uv-plane and compare directly to the visiblities. We use a non-uniform fast Fourier 
transform, which is the most efficient method for interferometer datasets containing ~1-10 million visibilities.
"""
settings_masked_interferometer = al.SettingsMaskedInterferometer(
    transformer_class=al.TransformerNUFFT
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    visibilities_mask=visibilities_mask,
    real_space_mask=real_space_mask,
    settings=settings_masked_interferometer,
)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` with `ExternalShear` [7 parameters].
 - An `EllipticalSersic` `LightProfile` for the source galaxy's light [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

NOTE: 

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllipticalSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/mass_sie__source_sersic/mass[sie]_source[bulge]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("interferometer", dataset_name),
    name="mass[sie]_source[inversion]",
    n_live_points=50,
)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `MaskedInterferometer`dataset.

For interferometer model-fits, we include a `SettingsInversion` object which describes how the linear algebra 
calculations required to use an `Inversion` are performed. One of two different approaches can be used: 

 - **Matrices:** Use a numerically more accurate matrix formalism to perform the linear algebra. For datasets 
 of < 100 0000 visibilities this approach is computationally feasible, and if your dataset is this small we we recommend 
 that you use this option (by setting `use_linear_operators=False`. However, larger visibility datasets these matrices 
 require excessive amounts of memory (> 16 GB) to store, making this approach unfeasible. 

 - **Linear Operators (default)**: These are slightly less accurate, but do not require excessive amounts of memory to 
 store the linear algebra calculations. For any dataset with > 1 million visibilities this is the only viable approach 
 to perform lens modeling efficiently.
"""
settings_inversion = al.SettingsInversion(use_linear_operators=True)

analysis = al.AnalysisInterferometer(
    dataset=masked_interferometer, settings_inversion=settings_inversion
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autolens_workspace/output/interferometer/mass_sie__source_sersic/mass[sie]_source[inversion]` for 
live outputs of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)


"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitInterferometer` objects.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=real_space_mask.masked_grid_sub_1
)
tracer_plotter.subplot_tracer()

fit_interferometer_plotter = aplt.FitInterferometerPlotter(
    fit=result.max_log_likelihood_fit
)
fit_interferometer_plotter.subplot_fit_interferometer()

"""
Checkout `autolens_workspace/notebooks/interferometer/modeling/results.py` for a full description of the result object.
"""
