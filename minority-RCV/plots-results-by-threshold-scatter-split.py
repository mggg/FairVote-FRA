
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

                    # Add some variance so we can split the data.
                    offset = 1/8 if ensembletype == "tilted" else -1/8
                    S = sum(results.flat)
                    
                    sigma = 1/32
                    xadj = np.random.normal(loc=0, scale=1/32)
                    yadj = np.random.normal(loc=0, scale=1/32)

                    # Create the point.
                    pt = (S+xadj+offset, thresholds+yadj)

                    if ensembletype == "tilted": tiltedpoints.append(pt)
                    else: neutralpoints.append(pt)


# Create subplots.
fig, ax = plt.subplots()

# Plot things!
ax.scatter(*zip(*neutralpoints), s=1/16, color="mediumpurple")
ax.scatter(*zip(*tiltedpoints), s=1/16, color="mediumseagreen")

# Set axes equal.
ax.set_aspect("equal")

# Labels.
ax.set_ylabel("Thresholds Exceeded")
ax.set_xlabel("Model-Predicted POC Wins")

# To file!
output = Path(f"./output/figures/scatterplots/")
if not output.exists(): output.mkdir()

# fig.tight_layout()
# plt.show()
plt.savefig(output/f"{location.abbr}-scatter-split.png", dpi=600, bbox_inches="tight")
