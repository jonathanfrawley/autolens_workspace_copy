"""
Tutorial 1: Non-linear Search
=============================

In this example, we're going to find a lens model that provides a good fit to an image, without assuming any knowledge
of what the `correct` lens model is.

So, whats a `lens model`? It is the combination of `LightProfile`'s and `MassProfile`'s we use to represent a lens galaxy,
source galaxy and therefore the strong lens ray-tracing configuration (i.e. the `Tracer`..

To begin, we have to choose the parametrization of our lens model. We don't need to specify the values of its light
and mass profiles (e.g. the centre, einstein_radius, etc.) - only the profiles themselves. In this example,
we'll use the following lens model:

 1) A `SphericalIsothermal` Sphere (SIS) for the lens galaxy's mass.
 2) A `SphericalExponential` `LightProfile` for the source-`Galaxy`'s light.

I'll let you into a secret - this is the same lens model used to simulate the `Imaging` data we're going to fit and
we're going to infer the actual parameters I used!

So, how do we infer the light and `MassProfile` parameters that give a good fit to our data?

Well, we could randomly guess a lens model, corresponding to some random set of parameters. We could use this
lens model to create a `Tracer` and fit the `Imaging` with it, via a `FitImaging` object. We can quantify how good the
fit is using its log likelihood (recall chapter_1/tutorial_8). If we kept guessing lens models, eventually we`d find
one that provides a good fit (i.e. high log_likelihood) to the data!

It may sound surprising, but this is actually the basis of how lens modeling works. However, we can do a lot better
than random guessing. Instead, we track the log likelihood of our previous guesses and guess more models using
combinations of parameters that gave higher log_likelihood solutions previously. The idea is that if a set of parameters
provided a good fit to the data, another set with similar values probably will too.

This is called a `non-linear search` and its a fairly common problem faced by scientists. Over the next few tutorials,
we're going to really get our heads around the concept of a non-linear search - intuition which will prove crucial to
being a successful lens modeler.

An animation of a non-linear search fitting a lens model can be found on the following page on our readthedocs. Note
how the initial models that it fits give a poor fit to the data, but gradually improve as more iterations are performed.

 `https://pyautolens.readthedocs.io/en/latest/overview/modeling.html`

we're going to use a non-linear search called `Dynesty`. I highly recommend it, and find its great for
lens modeling. However, for now, lets not worry about the details of how Dynesty actually works. Instead, just
picture that a non-linear search in **PyAutoLens** operates as follows:

 1) Randomly guess a lens model and use its `LightProfile`'s and `MassProfile`'s to set up a lens galaxy, source galaxy
 and a `Tracer`.

 2) Use this `Tracer` and a `Imaging` to generate a model image and compare this model image to the
 observed strong lens `Imaging` data using a `FitImaging` object, providing the log likelihood.

 3) Repeat this many times, using the likelihoods of previous fits (typically those with a high log_likelihood) to
 guide us to the lens models with the highest log likelihood.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af  # <- This library is used for non-linear fitting.
import autolens as al
import autolens.plot as aplt

"""
Lets loads the `Imaging` dataset we'll fit a lens model with using a non-linear search. If you are interested in how
we simulate strong lens data, checkout the scripts in the folder `autolens_workspace/howtolens/simulators`.

The strong lens in this image was generated using:

 - The lens galaxy's total mass distribution is a `SphericalIsothermal`.
 - The source galaxy's `LightProfile` is a `SphericalExponential`.

This dataset (and all datasets used in tutorials from here are on) are stored and loaded from the 
`autolens_workspace/dataset/imaging` folder.
"""
dataset_name = "mass_sis__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
The non-linear fit also needs a `Mask2D`, lets use a 3.0" circle.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
To compose a lens model, we set up a `Galaxy` as a `Model`. Whereas previous, we manually specified the value of 
every parameter of a `Galaxy`'s `LightProfile`'s and  `MassProfile`'s, when it is a `Model` these only the class of each
profile is passed and its parameters are fitted for and inferred by the non-linear search.

Lets model the lens galaxy with an `SphericalIsothermal` `MassProfile`.(which is what it was simulated with).
"""
lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.SphericalIsothermal)

"""
Lets model the source galaxy with a spherical exponential `LightProfile` (again, what it was simulated with).
"""
source_galaxy_model = af.Model(
    al.Galaxy, redshift=1.0, bulge=al.lp.SphericalExponential
)

"""
We now have multiple `Model` components, which we bring together into a final model via the `Collection` object.

Just like we are used to giving profiles descriptive names, like `bulge`, `disk` and `mass` we also name the galaxies 
that make up our model. Of course, its good practise for us to give them descriptive names and we'll use `lens` and
`source` to do this throughout the tutorials.

[It may seem odd that we define two `Collections`, with the `Collection` in the outer loop only having a `galaxies`
attribute. In future tutorials, we'll find that we can add additional model-components to a model other than just
galaxies, and the API below therefore makes it simple to extend the model to include these components.]
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
)

"""
We now create the non-linear search object which will fit the lens model, which as discussed above is the nested
sampling algorithm dynesty. We pass the `DynestyStatic` object the following:
 
 - Input parameters like `n_live_points` and `walks` which control how it samples parameter space. we'll cover what 
   these do in a later tutorial.
   
 - A `path_prefix` which tells the search to output its results in the folder `autolens_workspace/output/howtolens/`. 
 
 - A `name`, which gives the search a name and means the full output path is 
   `autolens_workspace/output/howtolens/tutorial_1_non_linear_search`. 

"""
search = af.DynestyStatic(
    path_prefix="howtolens",
    name="tutorial_1_non_linear_search",
    n_live_points=40,
    walks=10,
)

"""
We next create the `AnalysisImaging` object which defines the `log_likelihood_function` used by the non-linear search 
to fit the model to the `Imaging`dataset.
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Model fits using a non-linear search can take a long time to run. Whilst the fit in this tutorial should take around 
~10 minutes, later tutorials will take upwards of hours! This is fine, afterall lens modeling is an inherently 
computationally expensive exercise, but does make going through these tutorials problematic.

Furthermore, in a Jupyter notebook, if you run the non-linear search you won't be able to continue the notebook until 
it has finished. For this reason, we recommend that you run the non-linear search in these tutorials not via your 
Jupyter notebook, but instead by running the tutorial script found in the 
`autolens_workspace/scripts/howtolens/chapter_2_lens_modeling` folder. 

This can be run either using the `python3 scripts/howtolens/chapter_2_lens_modeling/tutoial_1_non_linear_search.py` 
command on your command line or via your IDE (if you are using one).

The non-linear search outputs all results to your hard-disk, thus if it runs and finishes in the script, you can then
run the Jupyter notebook cell and immediately load the result.
"""
print(
    "Dynesty has begun running - checkout the autolens_workspace/output/"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("Dynesty has finished run - you may now continue the notebook.")

"""
Now this is running you should checkout the `autolens_workspace/output` folder.

This is where the results of the search are written to your hard-disk (in the `tutorial_1_non_linear_search` folder). 
When its  completed, images and output will also appear in this folder, meaning that you don't need to keep running 
Python code to see the result.

In fact, even when a search is running, it outputs the the current maximum log likelihood results of the lens model 
to your hard-disk, on-the-fly. If you navigate to the output/howtolens folder, even before the search has finished, 
you'll see:

 1) The `image` folder, where the current maximum log likelihood lens model `Tracer` and `FitImaging` are visualized 
 (again, this outputs on-the-fly).
 
 2) The file `samples/samples.csv`, which contains a table-format list of every sample of the non-linear search
 complete with log likelihood values.
 
 3) The `model.info` file, which lists all parameters of the lens model and their priors.
 
 4) The `model.results` file, which lists the current best-fit lens model (this outputs on-the-fly).
 
 5) The `output.log` file, where all Python interpreter output is directed.

The maximum log likelihood is stored in the `result`, which we can plot as per usual (you must wait for the non-linear 
search to finish before you can get the `result` variable). we'll discuss the  `result` returned by a search in 
detail at the end of the chapter.
"""
fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

"""
The fit looks good and we've therefore found a model close to the one I used to simulate the image with (you can 
confirm this yourself if you want, by comparing the inferred parameters to those found in the script
`autolens_workspace/notebooks/imaging/simulators/no_lens_light/mass_sis__source_sersic.py`).

And with that, we're done - you`ve successfully modeled your first strong lens with **PyAutoLens**! Before moving onto 
the next tutorial, I want you to think about the following:

 1) a non-linear search is often said to search a `non-linear parameter-space` - why is the term parameter-space 
 used?

 2) Why is this parameter space 'non-linear'?

 3) Initially, the non-linear search randomly guesses the values of the parameters. However, it shouldn`t `know` 
 what reasonable values for a parameter are. For example, it doesn`t know that a reasonable Einstein radius is 
 between 0.0" and 4.0"). How does it know what are reasonable values of parameters to guess?
"""
