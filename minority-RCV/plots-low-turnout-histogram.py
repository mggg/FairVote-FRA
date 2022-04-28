
# Conjoined boxplot-simple-greengreen plots.

#######################################
# Boxplot and simple seat projection. #
#######################################

import jsonlines
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from pathlib import Path
import us
import sys
import pandas as pd
import numpy as np
from evaltools.plotting import districtr

from ModelingResult import ModelingResult, aggregate


# Get the location for the chain and the bias. State the focus states.
location = us.states.lookup(sys.argv[-1].title(), field="name")
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
poc = pd.read_csv("./data/demographics/pocrepresentation.csv")
statewide = pd.read_csv("./data/demographics/summary.csv")

summary = statewide.merge(poc, on="STATE")\
    .set_index("STATE")\
    .to_dict("index")

state = summary[location.name.title()]

output = Path("./output/")
planpath = Path(output/f"results/{location.name.lower()}/")

ensembletypes = ["", "-lowturnout"]

# Are we doing reduced turnout?
reducedturnout = True

ensembles = []

for ensembletype in ensembletypes:
    # Bucket for plans.
    plans = []

    # Check to see whether we have the complete results.
    representatives = state["REPRESENTATIVES"]

    for plan in range(50 if location in focus else 5):
        districts = []

        # We only care about the tilted low-turnout ensemble, not the other one;
        # we need a comparison!
        try:
            with jsonlines.open(planpath/f"neutral-{plan}{ensembletype}.jsonl") as r:
                for district in r:
                    districts.append([ModelingResult(**c) for c in district])
        except: continue
        plans.append(districts)

    ensembles.append(plans)

# Name the models.
models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]
concentrations = ["A", "B", "C", "D"]

# Subsample size.
subsample = 3

# Create ensembles!
neutralseats, lowturnoutseats = [], []
aggregatable = [neutralseats, lowturnoutseats]

for ensemble, aggregated in zip(ensembles, aggregatable):
    aggregated.extend(
        aggregate(ensemble, models, concentrations, subsample=subsample)["all"]
    )

# Set colors.
colors = districtr(4)
neutralcolor, lowturnoutcolor = colors[2], colors[0]

# Create figures.
fig, greengreen = plt.subplots(figsize=(7, 3))

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

# Method for computing y-ticks.
def computeyticks(ax, n, t):
    # Get the maximum y-value for the plot and the total number of observations.
    ymax = max(max(n[1]), max(t[1]))
    total = sum(n[1])

    # Set the interval to be 10%, then find the greatest number of intervals
    # exceeded (plus one); we then set the y maximum to be this value.
    interval = total*(1/10)
    mostintervals = np.ceil(ymax/interval)
    ymax = int(mostintervals*interval)

    # Now, create ticks appropriately.
    ticks = range(0, total+1, int(interval))
    greengreen.set_yticks(ticks, [str(round((t/total)*100))+"%" for t in ticks])

    # Set labels afterward so we crop the plot appropriately.
    greengreen.set_ylim(0, ymax)

# Also do one for x-ticks.
def computexticks(ax):
    ticks = greengreen.get_xticks()
    xmin, xmax = greengreen.get_xlim()

    if max(ticks)-min(ticks) > 10: ticks = list(np.arange(xmin, xmax+2))[::2]
    else: ticks = np.arange(xmin, xmax+1)

    greengreen.set_xticks(ticks, [str(int(t)) if t <= state["REPRESENTATIVES"] else "" for t in ticks])

#############################################
# Only plot --- histogram of districts won. #
#############################################

# Count the occurrences.
ns = list(np.unique(neutralseats, return_counts=True))
ts = list(np.unique(lowturnoutseats, return_counts=True))

# Adjust the things slightly.
ns[0] = [t-1/8 for t in ns[0]]
ts[0] = [t+1/8 for t in ts[0]]

# Set defaults.
defaults = dict(edgecolor="k", align="center", linewidth=1, alpha=3/4, width=3/8)

# Plot the histograms against each other.1)
greengreen.bar(*ns, color=neutralcolor, **defaults)
greengreen.bar(*ts, color=lowturnoutcolor, **defaults)

# Set axis label and title!
greengreen.set_ylabel("Frequency", fontdict={"fontsize": 8, **lfd})
greengreen.set_xlabel("Statewide seats", fontdict={"fontsize": 8, **lfd})
# greengreen.set_title("Detailed simulations", fontdict={"fontsize": 10, **lfd})

# Set plot limits!
xmin = min(min(neutralseats), min(lowturnoutseats))
xmax = max(max(neutralseats), max(lowturnoutseats))
greengreen.set_xlim(xmin-1 if xmin-1 > 0 else 0, xmax+1)

# Set y-ticks!
computeyticks(greengreen, ns, ts)
computexticks(greengreen)

# Fix labels on both plots.
for tickset in [greengreen.get_yticklabels(), greengreen.get_xticklabels()]:
    for tick in tickset: tick.set_fontproperties(fp)

# Create a legend.
handledefs = dict(edgecolor="k", linewidth=1, alpha=1/2)
handles = [
    Patch(facecolor=neutralcolor, label="Neutral", **handledefs),
    Patch(facecolor=lowturnoutcolor, label="Low-turnout", **handledefs)
]

greengreen.legend(
    prop=FontProperties(family="Playfair Display", size=8), title="Detailed seat projections",
    borderaxespad=0, bbox_to_anchor=(0.5, 1.03), ncol=2, handles=handles,
    title_fontproperties=FontProperties(family="Playfair Display", size=10), loc="lower center"
)

# To file!
output = Path(f"./output/figures/nationwide/")
if not output.exists(): output.mkdir()

# fig.tight_layout()
# plt.show()
figpath = Path(output/f"{location.abbr}-low-turnout.png")
