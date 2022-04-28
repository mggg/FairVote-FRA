
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from pathlib import Path
import us
import sys
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement as cwr
from evaltools.plotting import districtr

from ModelingResult import ModelingResult, aggregate

# Get the location for the chain and the bias.
location = us.states.lookup(sys.argv[-1].title(), field="name")
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
summary = pd.read_csv("./data/demographics/summary.csv")\
    .set_index("STATE")\
    .to_dict("index")

state = summary[location.name.title()]

# Load the plans for the ensemble.
output = Path("./output/")
planpath = Path(output/f"results/{location.name.lower()}/")

if location in focus:
    ensembletypes = ["neutral", "tilted"]
    tilted = True
else:
    ensembletypes = ["neutral"]
    tilted = False

# Bucket for ensembles.
ensembles = []

for ensembletype in ensembletypes:
    # Bucket for plans.
    plans = []

    # Check to see whether we have the complete results.
    tpath = planpath/f"{ensembletype}-0.jsonl"
    representatives = state["REPRESENTATIVES"]

    if tpath.exists():
        for plan in range(5 if representatives > 5 else 1):

            districts = []
            with jsonlines.open(planpath/f"{ensembletype}-{plan}.jsonl") as r:
                for district in r:
                    districts.append([ModelingResult(**c) for c in district])
            plans.append(districts)
    else:
        with jsonlines.open(planpath/f"{ensembletype}.jsonl") as r:
            for plan in r:
                districts = []
                for district in plan.values():
                    districts.append([ModelingResult(**c) for c in district])
                plans.append(districts)

        # Cascade!
        plans.append(districts)
    ensembles.append(plans)

models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]
concentrations = ["A", "B", "C", "D"]
pools = [1, 2, 3, 4]

# Sample size.
subsample = 3

# Aggregate results.
neutralresults = aggregate(ensembles[0], models, concentrations, subsample=subsample)
tiltedresults = aggregate(ensembles[1], models, concentrations, subsample=subsample)

# Merge the dictionaries!
models = {
    model: neutralresults[model] + tiltedresults[model]
    for model in models + ["all"]
}

# Counting!
modelresults = {
    name: {
        seats: count/len(model)
        for seats, count in dict(Counter(model)).items()
    }
    for name, model in models.items()
}

# Get everything.
modelresults["alt. crossover"] = modelresults["crossover"]
modelresults["Combined"] = modelresults["all"]
del modelresults["all"]
del modelresults["crossover"]

# Create plots.
fig, axes = plt.subplots(5, 1)
pl, bt, ac, cs, comb = axes

# Adjust subplot spacing.
fig.subplots_adjust(hspace=0.0)

# Set some defaults.
bardefs = dict(
    width=3/8, linewidth=1, edgecolor="k"
)

# Set colors.
colors = districtr(5)

# Set ymax, xmax, xmin.
ymax = 0
xmax = 0
xmin = 10000

models = ["plackett-luce", "bradley-terry", "alt. crossover", "cambridge", "Combined"]

for model, ax, color in zip(models, axes, colors):
    seats, shares = modelresults[model].keys(), modelresults[model].values()

    ax.bar(seats, shares, color=color, label=model.title(), **bardefs)
    # ax.bar(combinedseats, combinedshares, color=combined, **bardefs)

    # Set proportionality line and Biden support line!
    ax.axvline(state["POCVAP20SEATS"], color="k", alpha=1/2)
    ax.axvline(
        state["REPRESENTATIVES"]*(state["2020_PRES_D"]/(state["2020_PRES_D"]+state["2020_PRES_R"])),
        color="darkorange", alpha=1/2
    )

    xmax = max(max(seats), xmax)
    xmin = min(min(seats), xmin)

# Make the x limits the same for each plot, make all the x- and y-ticks (and 0-labels)
# invisible, and adjust the y limits to make the plots a tad taller.
for ax in axes:
    ax.set_xlim(xmin-1 if xmin%2 else xmin-2, xmax+1 if xmax%2 else xmax+2)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

# Now, for the last one, do x-ticks, and make those ticks pretty.
comb.axes.get_xaxis().set_visible(True)
for tick in comb.get_xticklabels(): tick.set_fontproperties(fp)

# Set the x-label for the last one.
comb.set_xlabel("Statewide seats", fontdict={**lfd, "fontsize": 10})

# Now create a legend.
handledefs = dict(edgecolor="k", linewidth=1)
handles = [
    Patch(facecolor=color, label=model.title(), **handledefs)
    for model, color in zip(modelresults.keys(), colors)
]

fig.legend(
    prop=FontProperties(family="Playfair Display", size=8), title="Detailed seat projection",
    borderaxespad=0, ncol=3, bbox_to_anchor=(0.5, 0.9),
    title_fontproperties=FontProperties(family="Playfair Display", size=10), loc="lower center"
)

figpath = Path(f"./output/figures/nationwide/{location.abbr}-plots-by-model-combined-histogram.png")
# fig.tight_layout()
# plt.show()
plt.savefig(figpath, dpi=600, bbox_inches="tight")
