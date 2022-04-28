
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from pathlib import Path
import numpy as np
import sys
import us
from ModelingResult import ModelingResult

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
planpath = Path(output/f"results/{location.name.lower()}")

# Create an empty container for ensembles.
ensembles = []

with jsonlines.open(planpath/"neutral.jsonl") as r: neutral = [p for p in r]
ensembles.append(neutral)

if location in focus:
    with jsonlines.open(planpath/"tilted.jsonl") as r: tilted = [p for p in r]
    ensembles.append(tilted)
else: tilted = None

# Create figures and axes.
fig, ax = plt.subplots(figsize=(6.5, 3))

# Create things for storing averages.
bands = { 3: [], 4: [], 5: [] }

for ensembletype, ensemble in enumerate(ensembles):
    # Now, get all the 3, 4, and 5-member districts to categorize them.
    buckets = {
        3: [[] for _ in range(threes)],
        4: [[] for _ in range(fours)],
        5: [[] for _ in range(fives)]
    }

    # Each plan is a dictionary which maps a district's index to the list of configurations
    # corresponding to that district. From there, we just need to order the districts
    # in the plan by POCVAP share – since the POCVAP share of each district in the plan
    # is constant over all configurations, we can just sample the first one and then
    # order them.
    for plan in ensemble:
        for index, district in plan.items():
            plan[index] = [ModelingResult(**r) for r in district]

    # For each plan, get a sample and order the indices of the districts.
    for plan in ensemble:
        sample = { index: district[0] for index, district in plan.items() }

        # Get a subsample and order the indices.
        for N in [3, 4, 5]:
            subsample = { i: d for i, d in sample.items() if d.seats == N }
            sortedsubsample = { i: d for i, d in sorted(subsample.items(), key=lambda k: k[1].pocshare) }
            sortedsubindices = list(sortedsubsample.keys())

            for bucketindex, districtindex in enumerate(sortedsubindices):
                for district in plan[districtindex]:
                    buckets[N][bucketindex].extend([
                        p/district.seats for p in district.pocwins
                    ])

                    bands[N].append(district.pocshare)

    # Get the sequence of shares! This is a complicated one and it's annoying.
    seq = [s for magnitude in buckets.values() for s in magnitude]

    # Compute the positions of the violins.
    positions = list(range(1, 2*(threes+fours+fives)+1)) if tilted else list(range(1, threes+fours+fives+1))
    if tilted:
        positions = positions[1::2] if ensembletype else positions[::2]
        positions = [p+(-1/7 if ensembletype else 1/7) for p in positions]

    # Create default parameters.
    defaults = dict(
        widths=len(positions)/8 * 1/2,
        positions=positions
    )

    # Plotting!
    v = ax.violinplot(seq, **defaults)
    bodies, mins, maxes, cbars = v["bodies"], v["cmins"], v["cmaxes"], v["cbars"]

    # Change the colors of the bodies.
    for body in bodies:
        body.set_facecolor("mediumseagreen" if ensembletype == 0 else "mediumpurple")
        body.set_edgecolor("k")
        body.set_linewidth(1/2)
        body.set_alpha(1)

    # Make the bars go away completely.
    for component in [cbars]: component.set_edgecolor("None")
    for component in [mins, maxes]: component.set_edgecolor("None")

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

# Create the posts!
posts = [r+1/2 for l, r in list(edges.values())[:-1]]
for post in posts: ax.axvline(x=post, ymin=0, ymax=1, ls=":", alpha=1/2, color="k")


# Create the bands!
for bucket, percentages in bands.items():
    if not percentages: continue

    # Get the min and max for that bucket, then plot the bands.
    pmin, pmax = min(percentages), max(percentages)
    l, r = edges[bucket]
    plt.gca().add_patch(Rectangle(
        (l-1/2, pmin), (r-l)+1, pmax-pmin,
        color="gold", alpha=1/4, zorder=0
    ))

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
ax.set_ylabel("Seat Share", fontdict=lfd)

# To file!
fig.tight_layout()
# plt.show()
if not (output/f"figures/{location.name.lower()}").exists():
    (f"figures/{location.name.lower()}").mkdir()
plt.savefig(output/f"figures/{location.name.lower()}/seatshare-by-ensemble.png", dpi=600, bbox_inches="tight")
