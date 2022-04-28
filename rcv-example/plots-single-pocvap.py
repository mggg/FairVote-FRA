
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
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

with jsonlines.open(planpath/"neutral.jsonl") as r: neutral = [p for p in r]
ensembles.append(neutral)

if location in focus:
    with jsonlines.open(planpath/"tilted.jsonl") as r: tilted = [p for p in r]
    ensembles.append(tilted)
else: tilted = None

ensembles = list(zip(
    ["neutral", "tilted"] if tilted else ["neutral"],
    ensembles
))

# Create figures and axes.
fig, ax = plt.subplots(figsize=(6.5, 3))

# Create things for storing results.
buckets = {
    3: {"neutral": [[] for _ in range(threes)], "tilted": [[] for _ in range(threes)]},
    4: {"neutral": [[] for _ in range(fours)], "tilted": [[] for _ in range(fours)]},
    5: {"neutral": [[] for _ in range(fives)], "tilted": [[] for _ in range(fives)]}
}

# Magnitude counts.
magcounts = list(zip([3, 4, 5], [threes, fours, fives]))

for ensembletype, ensemble in ensembles:
    # Categorize based on magnitude.
    for plan in ensemble:
        for magnitude, count in magcounts:
            districtsofmag = list(sorted(d["POCVAP20%"] for d in plan.values() if d["MAGNITUDE"] == magnitude))

            # Add to the appropriate list.
            for i in range(count):
                buckets[magnitude][ensembletype][i].append(districtsofmag[i])

    # Compute the positions of the violins.
    positions = list(range(1, 2*(threes+fours+fives)+1)) if tilted else list(range(1, threes+fours+fives+1))
    if tilted:
        positions = positions[1::2] if ensembletype == "tilted" else positions[::2]
        positions = [p+(-1/7 if ensembletype == "tilted" else 1/7) for p in positions]

    # Get the sequence of results!
    seq = []

    for magnitude, count in magcounts:
        seq.extend(buckets[magnitude][ensembletype])

    # Get districtr colors.
    nc, tc = districtr(2)

    # Create default parameters.
    defaults = dict(
        widths=len(positions)/8 * 1/2,
        positions=positions,
        patch_artist=True,
        whis=(1, 99),
        showfliers=False,
        boxprops=dict(
            facecolor=nc if ensembletype == "neutral" else tc
        ),
        medianprops=dict(
            linewidth=1, color="k"
        )
    )

    # Plotting!
    b = ax.boxplot(seq, **defaults)

# From here, we need to compute where the "posts" are. There are a few config
# scenarios here based on the whether there are districts in each bucket. Then,
# adjust the edges for side-by-side plots.
rawedges = [
    (0, threes),
    (threes, threes+fours),
    (threes+fours, threes+fours+fives)
]

if tilted: rawedges = [(l*2, r*2) for (l, r) in rawedges]

edges = {
    bucket: (l+1, r)
    for bucket, (l, r) in zip([3, 4, 5], rawedges)
    if l != r
}

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
for label, loc in labels: ax.text(loc, -1/10, label, **labeldefaults)

# Take away x-tick values and x-tick markers.
ax.axes.get_xaxis().set_visible(False)

# Set y-tick labels.
ylocs = [l*0.1 for l in range(0, 11)]
ax.set_yticks(ylocs, [str(round(100*l))+"%" for l in ylocs])

# Set plot limits.
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(1/2, (len(positions)*2 if tilted else len(positions))+1/2)

# Now go through and set font properties? this is bullshit
for tickset in [ax.get_xticklabels(), ax.get_yticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

# Set y-label.
ax.set_ylabel("POCVAP Share", fontdict=lfd)

# To file!
fig.tight_layout()
if not (output/f"figures/{location.name.lower()}").exists():
    (f"figures/{location.name.lower()}").mkdir()
plt.savefig(output/f"figures/{location.name.lower()}/POCVAP-combined.png", dpi=600, bbox_inches="tight")