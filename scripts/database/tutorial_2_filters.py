"""
Database 2: Filters
===================

Lets suppose we had the results of other fits in the folder `output/aggregator`, and we *only* wanted fits which used
the search defined in `phase_runner.py`. To avoid loading all the other results, we can use the aggregator`s filter
tool, which filters the results and provides us with only the results we want.

The filter provides us with the aggregator object we used in the previous tutorial, so can be used in an identical
fashion to tutorial 1.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af

"""
First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator(directory=path.join("output", "database"))

"""
We can filter results to only include completed results. By including the `completed_only` input below, any 
results which are in the middle of a non-linear will be omitted and not loaded in the `Aggregator`.
"""
agg = af.Aggregator(directory=path.join("output", "database"), completed_only=True)

"""
We can filter using strings, requiring that the string appears in the full path of the output
results. This is useful if you fit a samples of lenses where:

 - Multiple results, corresponding to different pipelines, searches and model-fits are stored in the same path.
 - Different runs using different `SettingsPhase` and `SetupPipeline` are in the same path.
 - Fits using different non-linear searches, with different settings, are contained in the same path.

The example below shows us using the contains filter to get the results of all 3 lenses. The contains method
only requires that the string is in the path structure, thus we do not need to specify the full search name.
"""
agg_filter = agg.filter(agg.directory.contains("phase_runner"))
print("Directory Filtered NestedSampler Samples: \n")
print("Total Samples Objects = ", len(list(agg_filter.values("samples"))), "\n\n")

"""
If we filter based on the dataset name, we can load the results of just one of the three model-fits performed in 
the tutorial_0 search runner.
"""
agg_filter = agg.filter(agg.directory.contains("mass_sie__source_sersic__0"))
print("Directory Filtered NestedSampler Samples: \n")
print("Total Samples Objects = ", len(list(agg_filter.values("samples"))), "\n\n")

"""
If we filtered using an incorrect search name we would get no results:
"""
name = "phase__incorrect_name"
agg_filter_incorrect = agg.filter(agg.directory.contains("invalid_string"))
print("Incorrect Phase Name Filtered NestedSampler Samples: \n")
print(
    "Total Samples Objects = ",
    len(list(agg_filter_incorrect.values("samples"))),
    "\n\n",
)

"""
Filters can be combined to load precisely only the result that you want, below we use all the above filters to 
load only the results of the fit to the first lens in our sample.
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
