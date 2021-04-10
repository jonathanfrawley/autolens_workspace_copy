"""
Modeling: Mass Total + Source Inversion
=======================================

In this script, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is an `Inversion`.

An `Inversion` reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization` and `Regularization` objects and in this example we will
use their simplest forms, a `Rectangular` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.
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

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

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
__Positions__

This fit also uses the arc-second positions of the multiply imaged lensed source galaxy, which were drawn onto the
image via the GUI described in the file `autolens_workspace/notebooks/imaging/preprocess/gui/positions.py`.
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `Rectangular` pixelization with fixed resolution 30 x 30 pixels (0 parameters).
 
 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 
 
It is worth noting the `Pixelization` and `Regularization` use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 

NOTE: 

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/modeling/mass_sie__source_sersic/mass[sie]_source[inversion]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling", dataset_name),
    name="mass[sie]_source[inversion]",
    nlive=50,
)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging`dataset. 

_Position Thresholding:_

Unlike other example scripts, we also pass the `AnalysisImaging` object below the positions we loaded above, alongside 
a `SettingsLens` object with a `positions_threshold`.

This is because `Inversion`'s suffer a bias whereby they fit unphysical lens models where the source galaxy is 
reconstructed as a demagnified version of the lensed source. These are covered in more detail in chapter 4 
of **HowToLens**. 

To prevent these solutions biasing the model-fit we specify a `position_threshold` of 0.5", which requires that a 
mass model traces the four (y,x) coordinates specified by our positions (that correspond to the brightest regions of the 
lensed source) within 0.5" of one another in the source-plane, else the mass model is discarded and a new 
model is sampled. This removes the unphysical solutions that bias an `Inversion`. 

The threshold of 0.5" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. However, we only want the threshold to aid the non-linear with the choice of mass model, but not risk 
removing genuinely physical models.

Position thresholding is described in more detail in the 
script `autolens_workspace/notebooks/imaging/modeling/customize/positions.py`
"""
analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=positions,
    settings_lens=al.SettingsLens(positions_threshold=0.5),
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autolens_workspace/output/examples/mass[sie]_source[bulge]` for live outputs of the results of the
fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

"""
Checkout `autolens_workspace/notebooks/imaging/modeling/results.py` for a full description of the result object.
"""
