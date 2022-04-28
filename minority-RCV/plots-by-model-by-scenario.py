
import jsonlines
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from pathlib import Path
import us
import sys
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement as cwr

from ModelingResult import ModelingResult

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

ensembles = []

for ensembletype in ensembletypes:
    with jsonlines.open(planpath/f"{ensembletype}.jsonl") as r:
        ensembles.append([p for p in r])

# Each plan is a dictionary which maps a district's index to the list of configurations
# corresponding to that district. From there, we just need to order the districts
# in the plan by POCVAP share -- since the POCVAP share of each district in the plan
# is constant over all configurations, we can just sample the first one and then
# order them.
for ensemble in ensembles:
    for plan in ensemble:
        for index, district in plan.items():
            plan[index] = [ModelingResult(**r) for r in district]


# Now, for each of the plans, sort the results into different buckets based on
# model type.
scenariomap = {
    "A": [0.5, 0.5, 0.5, 0.5],
    "B": [2, 0.5, 0.5, 0.5],
    "C": [2, 2, 2, 2],
    "D": [0.5, 0.5, 2, 2]
}

A = { ensembletype: {"plackett-luce": [], "bradley-terry": [], "crossover": [], "cambridge": []} for ensembletype in ensembletypes }
B = { ensembletype: {"plackett-luce": [], "bradley-terry": [], "crossover": [], "cambridge": []} for ensembletype in ensembletypes }
C = { ensembletype: {"plackett-luce": [], "bradley-terry": [], "crossover": [], "cambridge": []} for ensembletype in ensembletypes }
D = { ensembletype: {"plackett-luce": [], "bradley-terry": [], "crossover": [], "cambridge": []} for ensembletype in ensembletypes }

models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]

scenarios = list(zip(
    [A, B, C, D],
    ["A", "B", "C", "D"]
))

for ensembletype, ensemble in zip(ensembletypes, ensembles):
    for plan in ensemble:
        for model in models:
            for scenario, name in scenarios:
                # Get the correct concentration.
                concentration = scenariomap[name]

                # For each district, get the configurations corresponding to the models,
                # for which there should be four: one for each candidate pool. Then,
                # we get the statewide seat totals for each configuration and each
                # simulation.
                statewide = {
                    id: [
                        c for c in configurations if c.model == model and c.concentration == concentration
                    ]
                    for id, configurations in plan.items()
                }

                for pool in range(4):
                    subtotals = np.array([
                        district[pool].pocwins for district in statewide.values()
                    ])

                    indices = list(cwr(range(len(subtotals[0])), r=len(subtotals)))

                    for indexset in indices:
                        i = np.array(indexset)
                        results = np.take_along_axis(subtotals, i[:,None], axis=1)
                        scenario[ensembletype][model].append(sum(results.flat))

# Now we want to figure out the *frequencies* of each seat total.
for scenario, name in scenarios:
    for ensemble in ensembletypes:
        for model in models:
            total = len(scenario[ensemble][model])
            scenario[ensemble][model] = {
                seats: count/len(scenario[ensemble][model])
                for seats, count in dict(Counter(scenario[ensemble][model])).items()
            }

# Create subplots.
fig, axes = plt.subplots(2, 2)

tl, tr, bl, br = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
modelplots = list(zip([tl, tr, bl, br], models))

# Plotting! Here, we want to make circles at the appropriate height by summing
# over the seat totals from the *plans*. Set some defaults, like the max radius
# of the circles.
r = 9/20
ymax, ymin = 0, 10000
cdefs = dict(
    linewidth=1/4,
    edgecolor="grey",
    zorder=1
)

for ax, model in modelplots:
    x = 1

    # Set proportionality line!
    ax.axhline(state["POCVAP20SEATS"], color="k", alpha=1/2, zorder=0)

    for scenario, name in scenarios:
        for ensemble in scenario.values():
            for y, share in ensemble[model].items():
                # Get the appropriate radius relative to the max radius.
                sr = r*np.sqrt(share)

                # Plot a circle!
                C = Circle((x, y), sr, facecolor=cm.PuBu(share), **cdefs)
                ax.add_patch(C)

                # Put a thing on top of the circle!
                # label = ax.text(x, y, str(int(share*100))+"%", ha="center", va="center", color="white", fontsize=8)
                # label.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="k")])
                if ymax < y: ymax = y
                if ymin > y: ymin = y
            
            x+=1
        x+=1
    
    # Set plot limits.
    ax.set_xlim(0, x-1)

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

# Now set all the plot mins/maxes.
for ax, model in modelplots:
    ax.set_ylim(ymin-1, ymax+1)

    # Set labels!
    pos = list(range(1, x)) if tilted else list(range(1, x))

    if tilted: labellocs = [np.mean([pos[i-1], pos[i]]) for i in range(1, len(pos), 3)]
    else: labellocs = [1/2, 5/2, 9/2, 13/2]

    # Don't name models for this one.
    """
    for (model, name), loc in zip([modelnames], labellocs):
        ax.text(
            loc+1/2, ymin-11/10,
            name.title(), rotation=60, fontsize=8, ha="right", va="top"
        )
    """

    # Take away x-tick values and x-tick markers.
    ax.axes.get_xaxis().set_visible(False)
    yticks = list(range(ymin-1, ymax+2))[::2]
    ax.set_yticks(yticks, [str(int(t)) if t >=0 else "" for t in yticks])

    # Now go through and set font properties? this is bullshit
    for tickset in [ax.get_yticklabels()]:
        for tick in tickset:
            tick.set_fontproperties(fp)

    # Set title.
    ax.set_title(model.title(), fontdict=lfd)

    # Set aspects equal.
    ax.set_aspect("equal")

# To file!
output = Path(f"./output/figures/{location.name.lower()}/")
if not output.exists(): output.mkdir()

fig.tight_layout()
# plt.show()
plt.savefig(output/"plots-by-model-by-scenario.png", dpi=600, bbox_inches="tight")
