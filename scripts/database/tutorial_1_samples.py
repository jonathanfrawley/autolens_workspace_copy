"""
Database 1: Samples
===================

After fitting a large suite of strong lens data, we can use the aggregator to load the results and manipulate,
interpret and visualize them using a Python script or Jupyter notebook.

This script uses the results generated by the script `/autolens_workspace/aggregator/phase_runner.py`, which fitted 3
simulated strong lenses with:

 - An `EllipticalIsothermal` `MassProfile` for the lens galaxy's mass.
 - An `EllipticalSersic` `LightProfile` for the source galaxy's light.

This fit was performed using one `PhaseImaging` object, and the first four tutorials (a1-a4) cover how to use the
aggregator on the results of `Phase`'s (as opposed to `Pipeline`'s). However, the aggregator API is extremely similar
across both and learning to use the aggregator with searches can be easily applied to the results of pipelines.

__Samples__

If you are familiar with the `Samples` object returned from a *PyAutoLens* model-fit (e.g. via a `Phase` or `Pipeline`)
You will be familiar with most of the content in this script. Nevertheless, the script also describes how to use
the `Aggregator`, so will be useful for you too!

__File Output__

The results of this fit are in the `autolens_workspace/output/aggregator` folder. First, take a look in this folder.
Provided you haven't rerun the runner, you`ll notice that all the results (e.g. samples, samples_backup,
model.results, images, etc.) are in .zip files as opposed to folders that can be instantly accessed.

This is because when the pipeline was run, the `remove_files` option in the `config/general.ini` was set to True.
This means all results (other than the .zip file) were removed. This feature is implemented because super-computers
often have a limit on the number of files allowed per user.

Bare in mind the fact that all results are in .zip files, we'll come back to this point in a second.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af

"""
To set up the aggregator we simply pass it the folder of the results we want to load.
"""
agg = af.Aggregator(directory=path.join("output", "database", "phase_runner"))

"""
Before we continue, take another look at the output folder. The .zip files containing results have now all been 
unzipped, such that the results are accessible on your laptop for navigation. This means you can run fits to many 
lenses on a super computer and easily unzip all the results on your computer afterwards via the aggregator.

To begin, let me quickly explain what a generator is in Python, for those unaware. A generator is an object that 
iterates over a function when it is called. The aggregator creates all objects as generators, rather than lists, or 
dictionaries, or whatever.

Why? Because lists store every entry in memory simultaneously. If you fit many lenses, you`ll have lots of results and 
therefore use a lot of memory. This will crash your laptop! On the other hand, a generator only stores the object in 
memory when it runs the function; it is free to overwrite it afterwards. Thus, your laptop won't crash!

There are two things to bare in mind with generators:

    1) A generator has no length, thus to determine how many entries of data it corresponds to you first must convert 
       it to a list.
    
    2) Once we use a generator, we cannot use it again and we'll need to remake it.

We can now create a `samples` generator of every fit, which creates `Sample`'s objects of our results. This object 
contains information on the result of the non-linear search.
"""
samples_gen = agg.values("samples")

"""
When we print this the length of this generator converted to a list of outputs we see 3 different NestSamples 
instances. These correspond to each fit of each phase to each of our 3 images.
"""
print("NestedSampler Samples: \n")
print(samples_gen)
print()
print("Total Samples Objects = ", len(list(samples_gen)), "\n")

"""
The `Samples` class contains all the parameter samples, which is a list of lists where:
 
 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
for samples in agg.values("samples"):

    print("All parameters of the very first sample")
    print(samples.parameters[0])
    print("The third parameter of the tenth sample")
    print(samples.parameters[9][2])

print("Samples: \n")
print(agg.values("samples"))
print()
print("Total Samples Objects = ", len(list(agg.values("samples"))), "\n")

"""
The `Samples` class contains the log likelihood, log prior, log posterior and weights of every sample, where:

   - The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
     normalization).
    
   - The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log
     posterior value.
      
   - The log posterior is log_likelihood + log_prior.
    
   - The weight gives information on how samples should be combined to estimate the posterior. The weight values 
     depend on the sampler used, for example for MCMC they will all be 1`s.

"""
for samples in agg.values("samples"):
    print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
    print(samples.log_likelihoods[9])
    print(samples.log_priors[9])
    print(samples.log_posteriors[9])
    print(samples.weights[9])

"""
We can use the outputs to create a list of the maximum log likelihood model of each fit to our three images.
"""
ml_vector = [samps.max_log_likelihood_vector for samps in agg.values("samples")]

print("Max Log Likelihood Model Parameter Lists: \n")
print(ml_vector, "\n\n")

"""
This provides us with lists of all model parameters. However, this isn't that much use, which values correspond to 
which parameters?

The list of parameter names are available as a property of the `Model` included with the `Samples`, as are labels 
which can be used for labeling figures.
"""
for samples in agg.values("samples"):
    model = samples.model
    print(model)
    print(model.parameter_names)
    print(model.parameter_labels)

"""
These lists will be used later for visualization, how it is often more useful to create the model instance of every fit.
"""
ml_instances = [samps.max_log_likelihood_instance for samps in agg.values("samples")]
print("Maximum Log Likelihood Model Instances: \n")
print(ml_instances, "\n")

"""
A model instance contains all the model components of our fit, most importantly the list of galaxies we specified in 
the pipeline.
"""
print(ml_instances[0].galaxies)
print(ml_instances[1].galaxies)
print(ml_instances[2].galaxies)

"""
These galaxies will be named according to the phase (in this case, `lens` and `source`).
"""
print(ml_instances[0].galaxies.lens)
print()
print(ml_instances[1].galaxies.source)

"""
Their `LightProfile`'s and `MassProfile`'s are also named according to the phase.
"""
print(ml_instances[0].galaxies.lens.mass)
print(ml_instances[1].galaxies.source.bulge)

"""
We can also access the `median pdf` model, which is the model computed by marginalizing over the samples of every 
parameter in 1D and taking the median of this PDF.
"""
mp_vector = [samps.median_pdf_vector for samps in agg.values("samples")]
mp_instances = [samps.median_pdf_instance for samps in agg.values("samples")]

print("Median PDF Model Parameter Lists: \n")
print(mp_vector, "\n")
print("Most probable Model Instances: \n")
print(mp_instances, "\n")
print(mp_instances[0].galaxies.lens.mass)
print()

"""
We can compute the model parameters at a given sigma value (e.g. at 3.0 sigma limits).

These parameter values do not account for covariance between the model. For example if two parameters are degenerate 
this will find their values from the degeneracy in the `same direction` (e.g. both will be positive). we'll cover
how to handle covariance in a later tutorial.

Here, I use "uv3" to signify this is an upper value at 3 sigma confidence,, and "lv3" for the lower value.
"""
uv3_vectors = [
    samps.vector_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]

uv3_instances = [
    samps.instance_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]

lv3_vectors = [
    samps.vector_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]

lv3_instances = [
    samps.instance_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]

print("Errors Lists: \n")
print(uv3_vectors, "\n")
print(lv3_vectors, "\n")
print("Errors Instances: \n")
print(uv3_instances, "\n")
print(lv3_instances, "\n")

"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

Here, "ue3" signifies the upper error at 3 sigma. 
"""
ue3_vectors = [
    samps.error_vector_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]

ue3_instances = [
    samps.error_instance_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]

le3_vectors = [
    samps.error_vector_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]
le3_instances = [
    samps.error_instance_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]

print("Errors Lists: \n")
print(ue3_vectors, "\n")
print(le3_vectors, "\n")
print("Errors Instances: \n")
print(ue3_instances, "\n")
print(le3_instances, "\n")

"""
The maximum log likelihood of each model fit and its Bayesian log evidence (estimated via the nested sampling 
algorithm) are also available.

Given each fit is to a different image, these are not very useful. However, in a later tutorial we'll look at using 
the aggregator for images that we fit with many different models and many different pipelines, in which case comparing 
the evidences allows us to perform Bayesian model comparison!
"""
print("Maximum Log Likelihoods and Log Evidences: \n")
print([max(samps.log_likelihoods) for samps in agg.values("samples")])
print([samps.log_evidence for samps in agg.values("samples")])

"""
We can also print the "model_results" of all searches, which is string that summarizes every fit`s lens model providing 
quick inspection of all results.
"""
results = agg.model_results
print("Model Results Summary: \n")
print(results, "\n")

"""
The Probability Density Functions (PDF's) of the results can be plotted using the library:

 corner.py: https://corner.readthedocs.io/en/latest/

(In built visualization for PDF's and non-linear searches is a future feature of PyAutoFit, but for now you`ll have to 
use the libraries yourself!).

(uncomment the code below to make a corner.py plot.)
"""
# import corner
#
# for samples in agg.values("samples"):
#
#     corner.corner(
#         xs=samples.parameters,
#         weights=samples.weights,
#         labels=samples.model.parameter_labels,
#     )

"""
Finished.
"""
