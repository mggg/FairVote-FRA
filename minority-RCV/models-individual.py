
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jsonlines
import numpy as np
from pathlib import Path
from collections import Counter


"""
Creates side-by-side typical STV outcomes based on candidate pool.
"""

# Specify the jurisdiction, from which we read both the simulation results.
JURIS = sys.argv[-2]
BIAS = sys.argv[-1]
RESULTS_PATH = Path(f"./data/fairvote/output/{JURIS}-{BIAS}.jsonl")
FIGURES_PATH = Path(f"./data/fairvote/figures/{JURIS}/")
if not FIGURES_PATH.exists(): FIGURES_PATH.mkdir()

# Read in the simulation results.
with jsonlines.open(RESULTS_PATH) as results: RESULT = list(results)[-1]

# For each of these results, we want to get the range of *all* the scenarios for
# each of the candidate pools and get the ranges. Create a bucket to store the
# data we need to make the plots.
POOLS = ["P2"]
BUCKET = { p: { k: {} for k in [3, 4, 5] } for p in POOLS }
SCENARIOS = ["A", "B", "C", "D"]
MODELS = ["cambridge"]

# What data do we care about?
GROUP = "POCVAP"
WHOLE = "VAP"
SEATS = "SEATS"
GROUPSHARE = GROUP + "%"

for pool in BUCKET:
    for magnitude in BUCKET[pool].keys():
        # Get the districts in the result of this magnitude and sort them by
        # group share. If there are none of the specified magnitude, move on;
        # otherwise, bucket them.
        districts = [d for d in RESULT.values() if d["facts"][SEATS] == magnitude]
        if not districts: continue

        # Sort the districts.
        districts = list(sorted(districts, key=lambda d: d["facts"][GROUPSHARE]))

        # For each of the districts, get proportions of seats won.
        for district in districts:
            district["PROPORTIONS"] = list(np.array([
                list(district[model][pool].values()) for model in MODELS
            ]).flat.copy() / district["facts"][SEATS])

        BUCKET[pool][magnitude] = [
            {       
                "PROPORTIONS": district["PROPORTIONS"],
                "GROUPSHARE": district["facts"][GROUPSHARE]
            }
            for district in districts
        ]


# Use LaTeX. Create figures and axes.
plt.rcParams.update({
    "text.usetex": True,
    # "text.latex.preamble": r'\usepackage[cm]{sfmath}',
    "font.family": "serif",
	"font.serif": "New Century Schoolbook"
})

# Set figure sizes.
figuresizes = {
    "maryland": (7, 3),
    "florida": (7, 3),
    "massachusetts": (7, 3),
    "texas": (7, 3),
    "illinois": (7, 3)
}

if len(POOLS) > 1: figs, axs = plt.subplots(len(POOLS))
else:
    figs, axs = plt.subplots(figsize=figuresizes[JURIS])
    figs = [figs]
    axs = [axs]

# Create an offset.
offset = 1/20
mins, maxes = [], []

# Create a split plot.
for pool, ax in zip(BUCKET, axs):
    # Turn the x-axis off.
    ax.axes.get_xaxis().set_visible(False)

    """
    # Plot an "axis" at -1.
    ax.vlines(0, ymin=0, ymax=1, color="k", capstyle="round")
    ax.hlines(1, xmin=-1/10, xmax=1/10, color="k", capstyle="round")
    ax.hlines(0, xmin=-1/10, xmax=1/10, color="k", capstyle="round")
    ax.text(-1/10, 1, r"$100\%$", ha="right", va="center")
    ax.text(-1/10, 0, r"$0\%$", ha="right", va="center")
    """

    # Plot the title.
    # ax.set_title(f"Candidate Pool {pool.split('P')[1]}")

    # First, figure out where we're putting dividing lines, and plot them.
    magnitudes = [m for m in BUCKET[pool] if len(BUCKET[pool][m])]
    widths = [len(BUCKET[pool][m]) for m in magnitudes]

    # Adjust the widths so they're the sum(s) of the widths before them. Additionally,
    # we no longer need the last width, only the last k-1 (where k is the number
    # of widths); this is just a fencepost issue. Also create a list of tuples
    # specifying the midpoints of the intervals (to plot magnitude indicators).
    posts = [
        w + (widths[i-1] if i else 0)
        for i, w in enumerate(widths)
    ]

    midpoints = [
        (posts[i-1] if i else 0) + (w/2 + (1/2 if w%2 else 0) if w > 1 else 1)
        for i, w in enumerate(widths)
    ]

    print(JURIS, midpoints)

    # For each of the posts, magnitudes, and midpoints, plot the posts and district
    # magnitude information.
    for i, (x, mag, mid) in enumerate(zip(posts, magnitudes, midpoints)):
        # Plot the post if it's not the last post.
        if i < len(posts)-1: ax.vlines(x+1/2, ymin=0, ymax=1, lw=3/4, alpha=1/2, ls=":", color="k")

        # Plot the magnitude of the districts.
        annotationfontsize = 10
        ax.text(mid, -0.06, f"{mag}-member", fontsize=annotationfontsize, ha="center")

    # Now, we just concat all the information from before!
    unflattened = [BUCKET[pool][m] for m in magnitudes if BUCKET[pool][m]]
    districts = [d for u in unflattened for d in u]

    # Set line widths.
    baselinewidth = 1
    seatsharewidth = 8
    pocvapmarkersize = 6

    seq = [district["PROPORTIONS"] for district in districts]
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(1/2, len(districts)+1/2)
    parts = ax.violinplot(seq, showmedians=True, widths=1/3*(len(districts)/8))

    # Change these colors to black.
    for part in ('cbars','cmins','cmaxes','cmedians'):
        body = parts[part]
        body.set_edgecolor("none")

    # Change these colors to the ones we talked about before.
    for body in parts["bodies"]:
        body.set_edgecolor("k")
        body.set_facecolor("#d12455")
        body.set_alpha(1)

    # Plot stuff!
    for x, district in enumerate(districts):
        # We're 1-indexing.
        x = x+1
        """
        # Plot base line.
        ax.vlines(x, ymin=0, ymax=1, color="k", lw=baselinewidth)
        ax.hlines(0, xmin=x-1/10, xmax=x+1/10, color="k", lw=baselinewidth)
        ax.hlines(1, xmin=x-1/10, xmax=x+1/10, color="k", lw=baselinewidth)

        seq.append(district["PROPORTIONS"])
        
        # Plot range if it's a range; plot a marker if it's a single thing.
        if len(Counter(district["PROPORTIONS"])) > 1:
            ax.vlines(
                x, ymin=min(district["PROPORTIONS"]), ymax=max(district["PROPORTIONS"]),
                color="#d12455", lw=seatsharewidth, capstyle="round"
            )
        else:
            ax.vlines(
                x, ymin=min(district["PROPORTIONS"])-offset/5,
                ymax=min(district["PROPORTIONS"])+offset/5, lw=seatsharewidth,
                capstyle="round", color="#d12455"
            )
        """

        # Plot group share.
        ax.plot(
            x, district["GROUPSHARE"], marker="s", color="#f0c93e",
            markersize=pocvapmarkersize, markeredgecolor="k", markeredgewidth=1/4
        )

# Set y-axis ticks.
locs, labels = plt.yticks()
labels = [str(int(loc*100)) + r"\%" for loc in locs]
plt.yticks(locs, labels)

# Create a legend.
handles = [
    Line2D([0], [0], color="#d12455", lw=4, solid_capstyle="round"),
    Line2D(
        [0], [0], color="None", marker="s", markersize=5,
        markerfacecolor="#f0c93e", markeredgewidth=1/4
    )
]

labels = [
    "Potential Seat Shares",
    f"{GROUP} Share"
]

plt.legend(
    handles=handles, labels=labels, loc="upper center", fontsize=10,
    bbox_to_anchor=(0.5, 1.2), borderaxespad=0, ncol=2
)

# Set plot-wide title.
# plt.suptitle(f"STV Simulations in {JURIS.capitalize()} (Per District)", fontsize=16, y=1.05)
# plt.show()
plt.savefig(FIGURES_PATH/f"{BIAS}-individual.png", bbox_inches="tight", dpi=600)
