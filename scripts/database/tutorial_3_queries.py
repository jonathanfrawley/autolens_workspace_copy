"""
Database 3: Queries
===================

Suppose we have the results of many fits in the `output` folder and we only wanted to load and inspect a specific set
of model-fits (e.g. the results of `tutorial_1_introduction`). We can use the database's querying tools to only load
the results we are interested in.

The database also supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset)
can be loaded.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al

"""
First, set up the aggregator as we did in the previous tutorial.
"""
# from autofit.database.aggregator import Aggregator
# database_file = path.join("output", "database", "database.sqlite")
# agg = Aggregator.from_database(path.join(database_file))
agg = af.Aggregator(directory=path.join("output", "database"))

"""
We can filter results to only include completed results. By including the `completed_only` input below, any 
results which are in the middle of a non-linear will be omitted and not loaded in the `Aggregator`.
"""
agg = af.Aggregator(directory=path.join("output", "database"), completed_only=True)

"""
We can use the `Aggregator`'s to query the database and return only specific fits that we are interested in. We first 
do this, using the `info` object, whereby we can query any of its entries, for example the `dataset_name` string we 
input into the model-fit above. 

By querying using the string `mass_sie__source_sersic__1` the model-fit to only the second strong lens is returned:
"""
# Feature Missing
# agg_query = agg.query(agg.directory.contains("mass_sie__source_sersic__1"))
# samples_gen = agg_query.values("samples")
agg_filter = agg.filter(agg.directory.contains("runner"))

"""
As expected, this list now has only 1 MCMCSamples corresponding to the second dataset.
"""
print("Directory Filtered NestedSampler Samples: \n")
print("Total Samples Objects = ", len(list(agg_filter.values("samples"))), "\n\n")

"""
If we query using an incorrect dataset name we get no results:
"""
# Feature Missing
# agg_query = agg.query(agg.directory.contains("invalid_string"))
# samples_gen = agg_query.values("samples")
agg_filter_incorrect = agg.filter(agg.directory.contains("invalid_string"))
print("Incorrect Phase Name Filtered NestedSampler Samples: \n")
print(
    "Total Samples Objects = ",
    len(list(agg_filter_incorrect.values("samples"))),
    "\n\n",
)

"""
We can also query based on the model fitted. 

For example, we can load all results which fitted an `EllipticalIsothermal` model-component, which in this simple 
example is all 3 model-fits.

The ability to query via the model is extremely powerful. It enables a user to fit many lens models to large samples 
of lenses efficiently load and inspect the results. 

[Note: the code `agg.galaxies.lens.mass` corresponds to the fact that in the `Model` we named the model components 
`galaxies`, `lens` and `mass`. If the `Model` had used a different name the code below would change correspondingly. 
Models with multiple galaxies are therefore easily accessed via the database.]
"""
mass = agg.galaxies.lens.mass
agg_query = agg.query(mass == al.mp.EllipticalIsothermal)
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects via `EllipticalIsothermal` model query = ",
    len(list(samples_gen)),
    "\n",
)

"""
Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the 
inferred value of `sersic_index` for the `EllipticalSersic` of the source's bulge is less than 3.0 (which returns only 
the first of the three model-fits).
"""
bulge = agg.galaxies.source.bulge
agg_query = agg.query(bulge.sersic_index < 3.0)
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects In Query `source.bulge.sersic_index < 3.0` = ",
    len(list(samples_gen)),
    "\n",
)

"""
Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted an `EllipticalIsothermal` mass model AND (using the & symbol) inferred a value of sersic index of 
less than 3.0 for the source's bulge. 

The OR logical clause is also supported via the symbol |.
"""
mass = agg.galaxies.lens.mass
bulge = agg.galaxies.source.bulge
agg_query = agg.query((mass == al.mp.EllipticalIsothermal) & (bulge.sersic_index < 3.0))
samples_gen = agg_query.values("samples")
print(
    "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ",
    len(list(samples_gen)),
    "\n",
)

"""
Old Aggregator API... delete soon.
"""
agg_filter_multiple = agg.filter(
    agg.directory.contains("phase__"),
    agg.directory.contains("dynesty"),
    agg.directory.contains("mass_sie__source_bulge__0"),
)
print("Multiple Filter NestedSampler Samples: \n")
print()
print(
    "Total Samples Objects = ", len(list(agg_filter_multiple.values("samples"))), "\n\n"
)

"""
Finished.
"""
