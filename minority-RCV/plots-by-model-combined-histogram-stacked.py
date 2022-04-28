
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
modelresults["Combined"] = modelresults["all"]

# Create plots.
fig, ax = plt.subplots()

# Set some defaults.
bardefs = dict(
    width=1/4, alpha=3/4, linewidth=1/2, edgecolor="k"
)

# Set floors and ceilings.
ymax = 0
xmax, xmin = 0, 10000
offset = 18/60

for model, color in zip(list(models.keys()) + ["Combined"], districtr(5)):
    seats, shares = list(modelresults[model].keys()), list(modelresults[model].values())
    seats = [s+offset for s in seats]
    offset -= 9/60

    ax.bar(seats, shares, color=color, label=model.title(), **bardefs)

    # Set ytick labels.
    ymax = max(max(shares), ymax)
    xmax = max(max(seats), xmax)
    xmin = min(min(seats), xmin)

# Set proportionality line and Biden support line!
ax.axvline(state["POCVAP20SEATS"], color="k", alpha=1/2)
ax.axvline(
    state["REPRESENTATIVES"]*(state["2020_PRES_D"]/(state["2020_PRES_D"]+state["2020_PRES_R"])),
    color="darkorange", alpha=1/2
)

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

yticks = np.arange(0, round(ymax, 1)+0.1 if ymax < 0.95 else 1, 0.1)
ylabels = [str(round(t*100)) + "%" for t in yticks]
ax.set_yticks(yticks, ylabels)

# Set xtick labels.
xticks = range(int(xmin)-1, int(xmax+2))[::2]
xlabels = [str(t) for t in xticks]
ax.set_xticks(xticks, xlabels)

# Set title and x-label.
ax.set_title("Detailed seat projection", fontdict=lfd)
ax.set_xlabel("Statewide seats", fontdict={**lfd, "fontsize": 10})

# Now go through and set font properties? this is bullshit
for tickset in [ax.get_xticklabels(), ax.get_yticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

# Create a legend.
handledefs = dict(edgecolor="grey", linewidth=1, alpha=3/4)
handles = [
    Patch(facecolor="lightcoral", label="Individual Model", **handledefs),
    Patch(facecolor="steelblue", label="Combined",  **handledefs)
]

ax.legend(
    prop=FontProperties(family="Playfair Display", size=8), title="Detailed Seat Projection",
    borderaxespad=0, ncol=3, bbox_to_anchor=(0.5, 1.08),
    title_fontproperties=FontProperties(family="Playfair Display", size=10), loc="lower center"
)

figpath = Path(f"./output/figures/nationwide/{location.abbr}-plots-by-model-combined-histogram-stacked.png")
fig.tight_layout()
plt.show()
# plt.savefig(figpath, dpi=600, bbox_inches="tight")
