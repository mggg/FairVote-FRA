
# Conjoined boxplot-simple-greengreen plots.

#######################################
# Boxplot and simple seat projection. #
#######################################

import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from pathlib import Path
from matplotlib.lines import Line2D
from evaltools.plotting import districtr
import numpy as np
import sys
import us

# Get the location for the chain and the bias. State the focus states.
location = us.states.lookup(sys.argv[-1].title(), field="name")
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
summary = pd.read_csv("./data/demographics/summary.csv")\
    .set_index("STATE")\
    .to_dict("index")

state = summary[location.name.title()]
threes, fours, fives = int(state["3"]), int(state["4"]), int(state["5"])

# Load the plans for the ensemble. If we're in a focus state, load both ensembles.
output = Path("../rcv-example/output/")
planpath = Path(output/f"records/{location.name.lower()}")

# Create an empty container for ensembles.
ensembles = []

# Read in the neutral ensemble data.
with jsonlines.open(planpath/"neutral.jsonl") as r: neutral = [p for p in r]
ensembles.append(neutral)
tilted = False

if location in focus:
    with jsonlines.open(planpath/"tilted.jsonl") as r: tdata = [p for p in r]
    ensembles.append(tdata)
    tilted=True

ensembles = list(zip(
    ["neutral", "tilted"] if tilted else ["neutral"],
    ensembles
))

# Create figures and axes.
fig = plt.figure(figsize=(7,3))
gs = GridSpec(1, 3, width_ratios=[1/4, 1/4, 1/2])
box, simple, greengreen = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# Create things for storing results.
buckets = {
    3: {"neutral": [[] for _ in range(threes)], "tilted": [[] for _ in range(threes)]},
    4: {"neutral": [[] for _ in range(fours)], "tilted": [[] for _ in range(fours)]},
    5: {"neutral": [[] for _ in range(fives)], "tilted": [[] for _ in range(fives)]}
}

# Magnitude counts.
magcounts = list(zip([3, 4, 5], [threes, fours, fives]))

# Getting the total number of thresholds crossed.
thresholds = {
    "neutral": [],
    "tilted": []
}

for ensembletype, ensemble in ensembles:
    # Categorize based on magnitude.
    for plan in ensemble:
        for magnitude, count in magcounts:
            districtsofmag = list(sorted(d["POCVAP20%"] for d in plan.values() if d["MAGNITUDE"] == magnitude))

            # Add to the appropriate list.
            for i in range(count):
                buckets[magnitude][ensembletype][i].append(districtsofmag[i])

        thresholds[ensembletype].append(sum(d["POCSEATS"] for d in plan.values()))      

    # Compute the positions of the violins.
    positions = list(range(1, threes+fours+fives+1))

    # Get the sequence of results!
    seq = []

    for magnitude, count in magcounts:
        seq.extend(buckets[magnitude][ensembletype])

    # Create default parameters.
    defaults = dict(
        positions=positions,
        patch_artist=True,
        whis=(1, 99),
        showfliers=False,
        boxprops=dict(
            facecolor="gainsboro" if ensembletype == "neutral" else "mediumpurple"
        ),
        medianprops=dict(
            linewidth=1, color="k"
        )
    )

    # Plotting!
    if ensembletype == "neutral": b = box.boxplot(seq, **defaults)

# From here, we need to compute where the "posts" are. There are a few config
# scenarios here based on the whether there are districts in each bucket. Then,
# adjust the edges for side-by-side plots.
rawedges = [
    (0, threes),
    (threes, threes+fours),
    (threes+fours, threes+fours+fives)
]

edges = {
    bucket: (l+1, r)
    for bucket, (l, r) in zip([3, 4, 5], rawedges)
    if l != r
}

# Get colors and create legend handles.
colors = ["b", "r", "y"]
handles = []

# Plot some threshold lines!
for bucket, (l, r) in edges.items():
    if l == r and l <= 1: l=0

    # Are we doing the pinstripe plots?
    pinstripes = True

    if pinstripes:
        # Get *all* the handles.
        for i in range(1, bucket+1):
            box.hlines(i/(bucket+1), l-1/2, r+1/2, color=colors[bucket-3], alpha=1/4)
    else:
        box.hlines(1/(bucket+1), l-1/2, r+1/2, color=colors[bucket-3], alpha=1/4)
    
    handles.append(Line2D([0],[0], color=colors[bucket-3], label=f"{bucket}-member thresholds"))

# Plot some x-ticks!
locs, ticks = [], []

for edge in rawedges:
    locs.extend(list(range(*edge)))
    ticks.extend([str(t) for t in range(1, len(locs)+1)])

# Create the posts!
posts = [r+1/2 for l, r in list(edges.values())[:-1]]
for post in posts: box.axvline(x=post, ymin=0, ymax=1, ls=":", alpha=1/2, color="k")

# Add a proportionality line!
box.axhline(state["POCVAP20%"], color="k", alpha=1/2)

# Create the labels!
labels = [
    (f"{k}-member", np.mean(range(int(l), int(r+1))))
    for k, (l, r) in edges.items()
]

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

labeldefaults = dict(ha="center", va="center", fontdict={"fontsize": 8, **lfd})
for label, loc in labels: box.text(loc, -1/10, label, **labeldefaults)

# Set y-tick labels.
ylocs = [l*0.1 for l in range(0, 11)]
box.set_yticks(ylocs, [str(round(100*l))+"%" for l in ylocs])

# Set plot limits.
box.set_ylim(0, 1)
box.set_xlim(1/2, len(positions)+1/2)

# Now go through and set font properties? this is bullshit
for tickset in [box.get_xticklabels(), box.get_yticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

# Set y-label and turn off x-axis labels.
box.set_ylabel("POCVAP share", fontdict=lfd)
box.get_xaxis().set_visible(False)

# Create a legend.
handles.append(Line2D([0],[0], color="k", alpha=1/2, label="POC proportionality"))
box.legend(
    prop=FontProperties(family="Playfair Display", size=7), title="Minority share across districts",
    borderaxespad=0, bbox_to_anchor=(0.5, 1.04), ncol=1, handles=handles,
    title_fontproperties=FontProperties(family="Playfair Display", size=9),
    loc="lower center"
)

# Now plot some histograms?
n = np.unique(thresholds["neutral"], return_counts=True)
t = np.unique(thresholds["tilted"], return_counts=True)

# Adjust neutral and tilted positions slightly.
if tilted:
    n = [[l-1/8 for l in n[0]] ,n[1]]
    t = [[l+1/8 for l in t[0]], t[1]]
else: n = n

colors = districtr(8)
neutralcolor, tiltedcolor = colors[2], colors[3]
histdefs = dict(edgecolor="k", align="center", linewidth=1)
simple.bar(*n, color=neutralcolor, width=3/8 if tilted else 1/2, **histdefs)
if tilted: simple.bar(*t, color=tiltedcolor, width=3/8, alpha=1/2, **histdefs)

# Set plot limits.
if tilted: xmin, xmax = int(min(thresholds["neutral"])), int(max(thresholds["tilted"]))
else: xmin, xmax = int(min(thresholds["neutral"])), int(max(thresholds["neutral"]))
ymax = int(max(max(n[1]), max(t[1]) if tilted else 0))

simple.set_xticks(range(xmin-1 if xmin-1>0 else 0, xmax+2))

# Fix y ticks.
simple.set_yticks([])
simple.set_yticklabels([])
"""yticks = simple.get_yticks()
simple.set_yticks(yticks, [str(int(int(l)/1000))+"%" for l in yticks])
simple.yaxis.tick_right()
simple.yaxis.set_label_position("right")"""

# Add x- and y-axis labels.
# simple.set_ylabel("Frequency", rotation=270, labelpad=15, fontdict=lfd)
simple.set_xlabel("Statewide seats", fontdict=lfd)
simple.set_title("Simple", fontdict={"fontsize": 9, **lfd})

# Create a legend.
handledefs = dict(edgecolor="k", linewidth=1)
handles = [
    Patch(facecolor=neutralcolor, label="Neutral", **handledefs)
] + ([Patch(facecolor=tiltedcolor, label="Optimized", alpha=1/2, **handledefs)] if tilted else []) \
+ [Line2D([0],[0], linewidth=1, color="red", label="Current POC representation")]

simple.legend(
    prop=FontProperties(family="Playfair Display", size=7), title="Seat projections",
    borderaxespad=0, bbox_to_anchor=(1.6, 1.12), ncol=3 if tilted else 2, handles=handles,
    title_fontproperties=FontProperties(family="Playfair Display", size=9), loc="lower center"
)

# Now go through and set font properties? this is bullshit
for tickset in [simple.get_yticklabels(), simple.get_xticklabels()]:
    for tick in tickset: tick.set_fontproperties(fp)


######################
# Green-green plots. #
######################

import jsonlines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from pathlib import Path
import us
import sys
import pandas as pd
import numpy as np
from evaltools.plotting import districtr

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

ensembles = []

# Create a mapping for configurations.
concentrations = {
    "A": [0.5]*4,               # Voters more or less agree on candidate order for each group.
    "B": [2,0.5,0.5,0.5],       # POC voters vary on POC candidate rank-order, but other groups agree.
    "C": [2]*4,                 # Voters don't agree on anything.
    "D": [0.5,0.5,2,2],         # POC voters agree, and white voters don't.
    # "E": [1]*4                  # No agreement or disagreement --- it's pandemonium.
}

for ensembletype in ensembletypes:
    # Bucket for plans.
    plans = []

    # Check to see whether we have the complete results.
    tpath = planpath/f"{ensembletype}-0.jsonl"
    representatives = state["REPRESENTATIVES"]

    if tpath.exists():
        # How many plans are we getting?
        total = 50 if location in focus else 5
        for plan in range(total if representatives > 5 else 1):

            districts = []
            with jsonlines.open(planpath/f"{ensembletype}-{plan}.jsonl") as r:
                for district in r:
                    C = [ModelingResult(**c) for c in district]

                    for c in C:
                        for name, concentration in concentrations.items():
                            if c.concentration == concentration:
                                c.concentrationname = name

                    districts.append(C)
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

# Name the models.
models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]
concentrations = ["A", "B", "C", "D"]

# Subsample size.
subsample = 3

# Create ensembles!
neutralseats, tiltedseats = [], []
aggregatable = [neutralseats, tiltedseats] if tilted else [neutralseats]

for ensemble, aggregated in zip(ensembles, aggregatable):
    aggregated.extend(
        aggregate(ensemble, models, concentrations, subsample=subsample)["all"]
    )


# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

# Method for computing y-ticks.
def computeyticks(ax, n, t):
    # Get the maximum y-value for the plot and the total number of observations.
    ymax = max(max(n[1]), max(t[1])) if tilted else max(n[1])
    total = sum(n[1])

    # Set the interval to be 10%, then find the greatest number of intervals
    # exceeded (plus one); we then set the y maximum to be this value.
    interval = total*(1/10)
    mostintervals = np.ceil(ymax/interval)
    ymax = int(mostintervals*interval)

    # Now, create ticks appropriately.
    ticks = range(0, total+1, int(interval))
    ax.set_yticks(ticks, [str(round((t/total)*100))+"%" for t in ticks])

    # Set labels afterward so we crop the plot appropriately.
    ax.set_ylim(0, ymax)

# Also do one for x-ticks.
def computexticks(ax):
    ticks = ax.get_xticks()
    xmin, xmax = ax.get_xlim()

    if 10 < max(ticks)-min(ticks) < 18: ticks = list(np.arange(xmin, xmax+2))[::2]
    elif 18 <= max(ticks)-min(ticks): ticks =  list(np.arange(xmin, xmax+2))[::3]
    else: ticks = np.arange(xmin, xmax+1)

    ax.set_xticks(ticks, [str(int(t)) if t <= state["REPRESENTATIVES"] else "" for t in ticks])

#############################################
# Only plot --- histogram of districts won. #
#############################################

# Count the occurrences.
ns = list(np.unique(neutralseats, return_counts=True))
ts = list(np.unique(tiltedseats, return_counts=True))

# Adjust the things slightly.
ns[0] = [t-1/8 for t in ns[0]]
ts[0] = [t+1/8 for t in ts[0]]

# Set defaults.
defaults = dict(edgecolor="k", align="center", linewidth=1, alpha=3/4, width=3/8)

# Plot the histograms against each other.1)
greengreen.bar(*ns, color=neutralcolor, **defaults)
if tilted: greengreen.bar(*ts, color=tiltedcolor, **defaults)


# Set axis label and title!
greengreen.set_ylabel("Frequency", rotation=270, labelpad=15, fontdict=lfd)
greengreen.set_xlabel("Statewide seats", fontdict=lfd)
greengreen.set_title("Detailed", fontdict={"fontsize": 9, **lfd})

# Set plot limits!
xmin = min(min(neutralseats), min(tiltedseats)) if tilted else min(neutralseats)
xmax = max(max(neutralseats), max(tiltedseats)) if tilted else max(neutralseats)
greengreen.set_xlim(xmin-1 if xmin-1 > 0 else 0, xmax+1)

# Add a line demarcating the number of POC-chosen representatives statewide.
greengreen.axvline(state["POCREPRESENTATIVES"], color="red")

# Set y-ticks!
greengreen.set_yticks([])
greengreen.set_yticklabels([])
# computeyticks(greengreen, ns, ts)
computexticks(greengreen)

# Change ticks to the right side.
greengreen.yaxis.tick_right()
greengreen.yaxis.set_label_position("right")

# Fix labels on both plots.
for tickset in [greengreen.get_yticklabels(), greengreen.get_xticklabels()]:
    for tick in tickset: tick.set_fontproperties(fp)

# To file!
output = Path(f"./output/figures/nationwide/")
if not output.exists(): output.mkdir()

# Fix some subplot spacing.
plt.subplots_adjust(wspace=0.1)

# fig.tight_layout()
# plt.show()
suffix = "-pinstripe" if pinstripes else ""
figpath = Path(f"./output/figures/nationwide/{location.abbr}-conjoined-projections{suffix}.png")
plt.savefig(figpath, dpi=600, bbox_inches="tight")
