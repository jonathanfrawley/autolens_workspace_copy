"""
Modeling: Light Parametric + Mass Total + Source Parametric
===========================================================

In this script, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is a parametric `EllSersic` bulge and `EllExponential` disk.
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `EllSersic`.
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

"""
__Dataset__

Load and plot the strong lens dataset `light_sersic_exp__mass_sie__source_sersic` via .fits files, which we will fit with 
the lens model.
"""
dataset_name = "light_sersic_exp__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Masking__

The model-fit requires a `Mask2D` defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's light is a parametric `EllSersic` bulge and `EllExponential` disk, the centres of 
 which are aligned [11 parameters].
 
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

NOTE: 

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
bulge.centre = disk.centre

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=al.mp.EllIsothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/). We make the following changes to the Dynesty __Settings__:

 - Increase the number of live points, `nlive`, from the default value of 50 to 100. 
 - Increase the number of random walks per live point, `walks` from the default value of 5 to 10. 
 
These changes are motivated by the higher dimensionality non-linear parameter space that including the lens light 
creates, which requires more thorough sampling by the non-linear search.

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/mass_sie__source_sersic/mass[sie]_source[bulge]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=100,
    walks=10,
)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging`dataset.
"""
analysis = al.AnalysisImaging(dataset=imaging)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.
 - Information on the posterior as estimated by the `Dynesty` non-linear search. 
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

"""
Checkout `autolens_workspace/notebooks/imaging/modeling/results.py` for a full description of the result object.
"""
