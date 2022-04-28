
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from pathlib import Path
import math
import us
import sys
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement as cwr
from collections import Counter

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
    # Bucket for plans.
    plans = []

    # Check to see whether we have the complete results.
    representatives = state["REPRESENTATIVES"]

    for plan in range(50):
        districts = []
        with jsonlines.open(planpath/f"{ensembletype}-{plan}.jsonl") as r:
            for district in r:
                districts.append([ModelingResult(**c) for c in district])
        plans.append(districts)

    ensembles.append(plans)

# Buckets for results.
tiltedpoints = []
neutralpoints = []

# Name the models.
models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]

# Subsample size.
subsample = 3

# Each plan is a dictionary which maps a district's index to the list of configurations
# corresponding to that district. From there, we just need to order the districts
# in the plan by POCVAP share -- since the POCVAP share of each district in the plan
# is constant over all configurations, we can just sample the first one and then
# order them.
for ensembletype, ensemble in zip(ensembletypes, ensembles):
    for plan in ensemble:
        # Count the number of thresholds crossed.
        thresholds = 0

        # First, get the POCVAP shares for each of the districts and find out how
        # many thresholds are crossed.
        for district, configurations in enumerate(plan):
            first = configurations.pop()
            thresholds += math.floor(first.pocshare/(1/(first.seats+1)))
        
        for model in models:

            statewide = {
                id: np.random.choice([
                    c for c in district if c.model == model
                ], size=5, replace=False)
                for id, district in enumerate(plan)
            }

            # Get pool indices.
            pools = list(cwr(range(len(statewide[0])), r=len(statewide)))

            for poolindices in pools:
                subtotals = np.array([
                    district[pool].pocwins[:subsample] for pool, district in zip(poolindices, statewide.values())
                ])

                indices = list(cwr(range(len(subtotals[0])), r=len(subtotals)))

                for indexset in indices:
                    i = np.array(indexset)
                    results = np.take_along_axis(subtotals, i[:,None], axis=1)
                    pt = (sum(results.flat), thresholds)

                    if ensembletype == "tilted": tiltedpoints.append(pt)
                    else: neutralpoints.append(pt)


# Create subplots.
fig, ax = plt.subplots()

# Count the occurrences.
nc = dict(Counter(neutralpoints))
tc = dict(Counter(tiltedpoints))

# Get the total number of observations.
T = len(neutralpoints) + len(tiltedpoints)

# Set the max radius, and find the smallest/largest x and y points.
r = 7/16
minx, maxx = 10000, 0
miny, maxy = 10000, 0

# Now, create pie charts for each thing!
for point in set(nc.keys()) | set(tc.keys()):
    # Find min/max.
    x, y = point

    if x < minx: minx = x
    if x > maxx: maxx = x
    if y < miny: miny = y
    if y > maxy: maxy = y

    # Try to get the results from both places and plot each!
    nr = nc.get(point, 0)
    tr = tc.get(point, 0)

    # Get the total number of results, and find the share of each, setting their
    # angle measures.
    t = nr+tr
    ns = (nr/t)*360

    # Now, we want to get the radius of the arc we'll be plotting.
    rs = r*np.sqrt(t/T)

    # Plot an arc for each ensemble's results!
    if np.isclose(ns, 360):
        a = Wedge(point, rs, 0, 360, color="mediumpurple")
        ax.add_patch(a)
    elif np.isclose(ns, 0):
        b = Wedge(point, rs, 0, 360, color="mediumseagreen")
        ax.add_patch(b)
    else:
        a = Wedge(point, rs, 0, ns, color="mediumpurple")
        b = Wedge(point, rs, ns, 360, color="mediumseagreen")
        ax.add_patch(a)
        ax.add_patch(b)

# Set aspect equal.
ax.set_aspect("equal")

# Set limits.
ax.set_xlim(minx-1, maxx+1)
ax.set_ylim(miny-1, maxy+1)

# Set y-ticks.
ylocs = range(miny-1, maxy+2)
ax.set_yticks(ylocs, [str(t) for t in ylocs])

ax.set_ylabel("Thresholds Exceeded")
ax.set_xlabel("Model-Predicted POC Wins")

# To file!
output = Path(f"./output/figures/scatterplots/")
if not output.exists(): output.mkdir()

fig.tight_layout()
# plt.show()
plt.savefig(output/f"{location.abbr}-pie-scaled.png", dpi=600, bbox_inches="tight")
