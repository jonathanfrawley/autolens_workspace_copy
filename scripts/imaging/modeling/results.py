"""
Modeling: Results
=================

After fitting strong lens data a search returns a `result` variable, which we have used sparingly throughout the
examples scripts to plot the maximum log likelihood tracer and fits. However, this `Result` object has a lot more
information than that, and this script will cover everything it contains.

This script uses the result generated in the script `autolens_workspace/notebooks/imaging/modeling/mass_total__source_parametric.py`.
If you have not run the script or its results are not present in the output folder, the model-fit will be performed
again to create the results.

This model-fit fits the strong lens `Imaging` data with:

 - An `EllIsothermal` `MassProfile` for the lens galaxy's mass.
 - An `EllSersic` `LightProfile` for the source galaxy's light.
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
The code below, which we have omitted comments from, reperforms all the tasks that create the search and perform the
model-fit in this script. If anything in this code is not clear to you, you should go over the beginner model-fit
script again.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=imaging)

result = search.fit(model=model, analysis=analysis)

"""
Great, so we have the `Result` object we'll cover in this script. As a reminder, we can use the 
`max_log_likelihood_tracer` and `max_log_likelihood_fit` to plot the results of the fit:
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid_sub_1
)
tracer_plotter.subplot_tracer()
fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

"""
The result contains a lot more information about the model-fit. 

For example, its `Samples` object contains the complete set of non-linear search samples, for example every set of 
parameters evaluated, their log likelihoods and so on, which are used for computing information about the model-fit 
such as the error on every parameter. Our model-fit used the nested sampling algorithm Dynesty, so the `Samples` object
returned is a `NestSamples` objct.
"""
samples = result.samples

print("Nest Samples: \n")
print(samples)

"""
The `Samples` class contains all the parameter samples, which is a list of lists where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
print("All parameters of the very first sample")
print(samples.parameter_lists[0])
print("The fourth parameter of the tenth sample")
print(samples.parameter_lists[9][3])

"""
The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every sample, where:

   - The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
     normalization).

   - The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log
     posterior value.

   - The log posterior is log_likelihood + log_prior.

   - The weight gives information on how samples should be combined to estimate the posterior. The weight values 
     depend on the sampler used. For example for an MCMC search they will all be 1`s whereas for the nested sampling
     method used in this example they are weighted as a combination of the log likelihood value and prior..
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
The `Samples` contain the maximum log likelihood model of the fit (we actually used this when we used the 
max_log_likelihood_tracer and max_log_likelihood_fit properties of the results).
"""
ml_vector = samples.max_log_likelihood_vector
print("Max Log Likelihood Model Parameters: \n")
print(ml_vector, "\n\n")

"""
This provides us with a list of all model parameters. However, this isn't that much use, which values correspond to 
which parameters?

The list of parameter names are available as a property of the `Samples`, as are parameter labels which can be used 
for labeling figures.
"""
print(samples.model.model_component_and_parameter_names)
print(samples.model.parameter_labels)

"""
These lists will be used later for visualization, however it can be more useful to create the model instance of every 
fit.
"""
ml_instance = samples.max_log_likelihood_instance
print("Maximum Log Likelihood Model Instance: \n")
print(ml_instance, "\n")

"""
A model instance contains all the model components of our fit, most importantly the list of galaxies we specified in 
the pipeline.
"""
print(ml_instance.galaxies)

"""These galaxies will be named according to the search (in this case, `lens` and `source`)."""
print(ml_instance.galaxies.lens)
print(ml_instance.galaxies.source)

"""Their `LightProfile`'s and `MassProfile`'s are also named according to the search."""
print(ml_instance.galaxies.lens.mass)

"""
We can use this list of galaxies to create the maximum log likelihood `Tracer`, which, funnily enough, 
is the property of the result we've used up to now!

(If we had the `Imaging` available we could easily use this to create the maximum log likelihood `FitImaging`.
"""
ml_tracer = al.Tracer.from_galaxies(galaxies=ml_instance.galaxies)

tracer_plotter = aplt.TracerPlotter(tracer=ml_tracer, grid=mask.unmasked_grid_sub_1)
tracer_plotter.subplot_tracer()

"""
We can also access the `median pdf` model, which is the model computed by marginalizing over the samples of every 
parameter in 1D and taking the median of this PDF.
"""
mp_vector = samples.median_pdf_vector
mp_instance = samples.median_pdf_instance

print("Median PDF Model Parameter Lists: \n")
print(mp_vector, "\n")
print("Most probable Model Instances: \n")
print(mp_instance, "\n")
print(mp_instance.galaxies.lens.mass)
print()

"""
We can compute the model parameters at a given sigma value (e.g. at 3.0 sigma limits).

These parameter values do not account for covariance between the model. For example if two parameters are degenerate 
this will find their values from the degeneracy in the `same direction` (e.g. both will be positive). we'll cover
how to handle covariance elsewhere.

Here, I use "uv3" to signify this is an upper value at 3 sigma confidence,, and "lv3" for the lower value.
"""
uv3_vector = samples.vector_at_upper_sigma(sigma=3.0)
uv3_instance = samples.instance_at_upper_sigma(sigma=3.0)
lv3_vector = samples.vector_at_lower_sigma(sigma=3.0)
lv3_instance = samples.instance_at_lower_sigma(sigma=3.0)

print("Errors Lists: \n")
print(uv3_vector, "\n")
print(lv3_vector, "\n")
print("Errors Instances: \n")
print(uv3_instance, "\n")
print(lv3_instance, "\n")

"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

Here, "ue3" signifies the upper error at 3 sigma. 

( Need to fix bug, sigh).
"""
# ue3_vector = samples.error_vector_at_upper_sigma(sigma=3.0)
# ue3_instance = samples.error_instance_at_upper_sigma(sigma=3.0)
# le3_vector = samples.error_vector_at_lower_sigma(sigma=3.0)
# le3_instance = samples.error_instance_at_lower_sigma(sigma=3.0)
#
# print("Errors Lists: \n")
# print(ue3_vector, "\n")
# print(le3_vector, "\n")
# print("Errors Instances: \n")
# print(ue3_instance, "\n")
# print(le3_instance, "\n")

"""
The maximum log likelihood of each model fit and its Bayesian log evidence (estimated via the nested sampling 
algorithm) are also available.
"""
print("Maximum Log Likelihood and Log Evidence: \n")
print(max(samples.log_likelihood_list))
print(samples.log_evidence)

"""
The Probability Density Functions (PDF's) of the results can be plotted using Dynesty's in-built visualization tools, 
which are wrapped via the `DynestyPlotter` object.
"""
dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()


"""
__Aggregator__

Once a search has completed running, we have a set of results on our hard disk we manually inspect and analyse. 
Alternatively, we return the results from the search.run() method and manipulate them in a Python script, as we did
in this script.

However, imagine your dataset is large and consists of many images of strong lenses. You analyse each image 
individually using the same search, producing a large set of results on your hard disk corresponding to the full sample.
That will be a lot of paths and directories to navigate! At some point, there`ll be too many results for it to be
a sensible use of your time to analyse the results by sifting through the outputs on your hard disk.

PyAutoFit`s aggregator tool allows us to load results in a Python script or, more importantly, a Jupyter notebook. This
bypasses the need for us to run a search and can load the results of any number of lenses at once, allowing us to 
manipulate the results of extremely large lens samples!

If the `Aggregator`. sounds useful to you, then checkout the tutorials in the path:

 `autolens_workspace/advanced/aggregator`
"""
