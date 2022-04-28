
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
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
output = Path("./output/")
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
fig = plt.figure(figsize=(7, 3))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])
ax, hist = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])

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
        widths=len(positions)/8 * 1/2,
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
    if ensembletype == "neutral": b = ax.boxplot(seq, **defaults)

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
    pinstripes = False

    if pinstripes:
        # Get *all* the handles.
        for i in range(1, bucket+1):
            ax.hlines(i/(bucket+1), l-1/2, r+1/2, color=colors[bucket-3], alpha=1/4)
    else:
        ax.hlines(1/(bucket+1), l-1/2, r+1/2, color=colors[bucket-3], alpha=1/4)
    
    handles.append(Line2D([0],[0], color=colors[bucket-3], label=f"{bucket}-member thresholds"))

# Plot some x-ticks!
locs, ticks = [], []

for edge in rawedges:
    locs.extend(list(range(*edge)))
    ticks.extend([str(t) for t in range(1, len(locs)+1)])

# Create the posts!
posts = [r+1/2 for l, r in list(edges.values())[:-1]]
for post in posts: ax.axvline(x=post, ymin=0, ymax=1, ls=":", alpha=1/2, color="k")

# Add a proportionality line!
ax.axhline(state["POCVAP20%"], color="k", alpha=1/2)

# Create the labels!
labels = [
    (f"{k}-member", np.mean(range(int(l), int(r+1))))
    for k, (l, r) in edges.items()
]

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

labeldefaults = dict(ha="center", va="center", fontdict=lfd)
for label, loc in labels: boxes.text(loc, -1/10, label, **labeldefaults)

# Take away x-tick values and x-tick markers.
ax.axes.get_xaxis().set_visible(False)

# Set y-tick labels.
ylocs = [l*0.1 for l in range(0, 11)]
ax.set_yticks(ylocs, [str(round(100*l))+"%" for l in ylocs])

# Set plot limits.
ax.set_ylim(0, 1)
ax.set_xlim(1/2, len(positions)+1/2)

# Now go through and set font properties? this is bullshit
for tickset in [ax.get_xticklabels(), ax.get_yticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

# Set y-label.
ax.set_ylabel("POCVAP share", fontdict=lfd)

# Create a legend.
handles.append(Line2D([0],[0], color="k", alpha=1/2, label="POC proportionality"))
ax.legend(
    prop=FontProperties(family="Playfair Display", size=7), title="Minority share across districts",
    borderaxespad=0, bbox_to_anchor=(0.5, 1.03), ncol=len(seq)+1, handles=handles,
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
nc, tc = colors[2], colors[3]
histdefs = dict(edgecolor="k", align="center", linewidth=1)
hist.bar(*n, color=nc, width=3/8 if tilted else 1/2, **histdefs)
if tilted: hist.bar(*t, color=tc, width=3/8, alpha=1/2, **histdefs)

# Set plot limits.
if tilted: xmin, xmax = int(min(thresholds["neutral"])), int(max(thresholds["tilted"]))
else: xmin, xmax = int(min(thresholds["neutral"])), int(max(thresholds["neutral"]))
ymax = int(max(max(n[1]), max(t[1]) if tilted else 0))

hist.set_xticks(range(xmin-1 if xmin-1>0 else 0, xmax+2))
hist.set_ylim(0, ymax)

# Fix y ticks.
yticks = hist.get_yticks()
hist.set_yticks(yticks, [str(int(int(l)/1000))+"%" for l in yticks])
hist.yaxis.tick_right()
hist.yaxis.set_label_position("right")

# Add x- and y-axis labels.
hist.set_ylabel("Frequency", rotation=270, labelpad=8, fontdict=lfd)
hist.set_xlabel("Statewide seats", fontdict=lfd)

# Create a legend.
handledefs = dict(edgecolor="k", linewidth=1)
handles = [
    Patch(facecolor=nc, label="Neutral", **handledefs)
] + ([Patch(facecolor=tc, label="Optimized", alpha=1/2, **handledefs)] if tilted else [])

hist.legend(
    prop=FontProperties(family="Playfair Display", size=7), title="Simple seat projection",
    borderaxespad=0, bbox_to_anchor=(0.5, 1.03), ncol=2 if tilted else 1, handles=handles,
    title_fontproperties=FontProperties(family="Playfair Display", size=9), loc="lower center"
)

# Now go through and set font properties? this is bullshit
for tickset in [hist.get_yticklabels(), hist.get_xticklabels()]:
    for tick in tickset: tick.set_fontproperties(fp)

# To file!
"""if location in focus:
    output = Path(f"./output/figures/{location.name.lower()}/")
    if not output.exists(): output.mkdir()
    figpath = output/"POCVAP-with-thresholds.png"
else:
    """
suffix = "-pinstripe" if pinstripes else ""
figpath = Path(f"./output/figures/nationwide/{location.abbr}-POCVAP-with-thresholds{suffix}.png")
plt.savefig(figpath, dpi=600, bbox_inches="tight")
