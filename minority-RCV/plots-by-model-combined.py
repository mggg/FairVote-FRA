
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from pathlib import Path
import us
import sys
import pandas as pd
import numpy as np
from collections import Counter

from ModelingResult import ModelingResult, aggregate

# Get the location for the chain and the bias.
location = us.states.lookup(sys.argv[-1].title(), field="name")
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
poc = pd.read_csv("./data/demographics/pocrepresentation.csv")
statewide = pd.read_csv("./data/demographics/summary.csv")

summary = statewide.merge(poc, on="STATE")\
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

# Create a mapping for configurations.
concentrations = {
    "A": [0.5]*4,               # Voters more or less agree on candidate order for each group.
    "B": [2,0.5,0.5,0.5],       # POC voters vary on POC candidate rank-order, but other groups agree.
    "C": [2]*4,                 # Voters don't agree on anything.
    "D": [0.5,0.5,2,2],         # POC voters agree, and white voters don't.
    # "E": [1]*4                  # No agreement or disagreement --- it's pandemonium.
}


# Bucket for ensembles.
ensembles = []

for ensembletype in ensembletypes:
    # Bucket for plans.
    plans = []

    # Check to see whether we have the complete results.
    tpath = planpath/f"{ensembletype}-0.jsonl"
    representatives = state["REPRESENTATIVES"]

    if tpath.exists():
        for plan in range(50 if representatives > 5 else 1):
            try:
                districts = []
                with jsonlines.open(planpath/f"{ensembletype}-{plan}.jsonl") as r:
                    for district in r:
                        C = [ModelingResult(**c) for c in district]

                        for c in C:
                            for name, concentration in concentrations.items():
                                if c.concentration == concentration:
                                    c.concentrationname = name

                        districts.append(C)
            except: continue
            plans.append(districts)
    else:
        with jsonlines.open(planpath/f"{ensembletype}.jsonl") as r:
            for plan in r:
                districts = []
                for district in plan.values():
                    C = [ModelingResult(**c) for c in district]

                    for c in C:
                        for name, concentration in concentrations.items():
                            if c.concentration == concentration: c.concentrationname = name
                    districts.append(C)
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
if tilted: tiltedresults = aggregate(ensembles[1], models, concentrations, subsample=subsample)

# Merge the dictionaries!
modelresults = {
    model: neutralresults[model] + (tiltedresults[model] if tilted else [])
    for model in models + ["all"]
}

# Counting!
modelresults = {
    name: {
        seats: count/len(model)
        for seats, count in dict(Counter(model)).items()
    }
    for name, model in modelresults.items()
}

# Get everything.
modelresults["alt. crossover"] = modelresults["crossover"]
modelresults["Combined"] = modelresults["all"]
del modelresults["all"]
del modelresults["crossover"]

modelresults = {
    model: modelresults[model]
    for model in ["plackett-luce", "bradley-terry", "alt. crossover", "cambridge", "Combined"]
}

# Create plots.
fig, ax = plt.subplots(figsize=(7,3))

# Plotting! Here, we want to make circles at the appropriate height by summing
# over the seat totals from the *plans*. Set some defaults, like the max radius
# of the circles.
r = 1/2
y = 1
ymax = 7
xmax, xmin = 0, 10000

# Some defaults for circles.
cdefs = dict(
    linewidth=3/4,
    edgecolor="grey",
    zorder=1
)

for name, model in modelresults.items():
    for x, share in model.items():
        # Get the appropriate radius relative to the max radius.
        sr = r*np.sqrt(share)

        # Plot a circle!
        color = "steelblue" if name == "Combined" else "mediumpurple"
        C = Circle((x, ymax-(y+1) if name == "Combined" else ymax-y), sr, facecolor=color)
        ax.add_patch(C)

        if xmax < x: xmax = x
        if xmin > x: xmin = x
    
    y+=1

# Set proportionality line and Biden support line!
ax.axvline(state["POCVAP20SEATS"], color="gold", alpha=1/2)
ax.axvline(
    state["REPRESENTATIVES"]*(state["2020_PRES_D"]/(state["2020_PRES_D"]+state["2020_PRES_R"])),
    color="k", alpha=1/2
)

# Re-set the xmin value if the number of POC representatives is smaller than everything!
# Really unlikely that the current number is bigger than everything.
xmin = min(xmin, state["POCREPRESENTATIVES"])

# Set labels!
pos = list(range(1, x+1))
labellocs = [1, 3, 4, 5, 6]

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

for m, loc in zip(modelresults.keys(), reversed(labellocs)):
    if m == "crossover": m = "alt. crossover"
    ax.text(
        xmin-6/5, loc+1/4, m.title(), fontsize=9, ha="right", va="top", fontdict=lfd
    )

# Put a break between the model results and the combined results.
# ax.axhline(5, color="k", alpha=1/2, ls=":")

# Take away x-tick values and x-tick markers.
ax.axes.get_yaxis().set_visible(False)

# Set plot limits.
ax.set_xlim(xmin-1, xmax+1)
ax.set_ylim(0, y+1)

xticks = list(range(xmin, xmax+1))
ax.set_xticks(
    xticks[::(1 if len(xticks) < 20 else 2)],
    [str(int(t)) if t >=0 else "" for t in xticks][::(1 if len(xticks) < 20 else 2)]
)

# Set aspects equal.
ax.set_aspect("equal")

# Now go through and set font properties? this is bullshit
for tickset in [ ax.get_xticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

# Set axis labels.
ax.set_xlabel("Statewide Seats", fontdict=lfd)

# Add a legend?
handles = [
    Line2D([0],[0], color="gold", alpha=1/2, label="POC proportionality"),
    Line2D([0],[0], color="k", alpha=1/2, label="Biden proportionality"),
    # Line2D([0],[0], color="forestgreen", alpha=1/2, label="Current POC representation")
]

ax.legend(
    prop=FontProperties(family="Playfair Display", size=7),
    title_fontproperties=FontProperties(family="Playfair Display", size=9),
    title="Detailed seat projection", handles=handles, ncol=2,
    borderaxespad=0, loc="lower center", bbox_to_anchor=(0.5, 1.05)
)

bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
width, height = bbox.width, bbox.height

# To file!
"""if location in focus:
    output = Path(f"./output/figures/{location.name.lower()}/")
    if not output.exists(): output.mkdir()
    figpath = output/"plots-by-model-combined.png"
else:
    """
figpath = Path(f"./output/figures/nationwide/{location.abbr}-plots-by-model-combined.png")
plt.savefig(figpath, dpi=600, bbox_inches="tight")
