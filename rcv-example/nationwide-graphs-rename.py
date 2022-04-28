
import os
from us import states
from pathlib import Path

# Set the graph root.
graphroot = Path("./data/graphs/")

# Set the list of exceptions.
exceptions = [states.OR, states.HI, states.CA]

for state in states.STATES:
    # Create the graph path; if the path exists and the state isn't in the list
    # of exceptions, keep the file and rename it; otherwise, delete it.
    graphpath = graphroot/f"{state.abbr.lower()}-vtd-connected.json"

    if state in exceptions:
        graphpath = graphroot/f"{state.abbr.lower()}-bg-connected.json"
        if not graphpath.exists(): continue

        graphpath.rename(graphroot/f"{state.name.lower()}.json")
        continue

    if not graphpath.exists(): continue
    graphpath.rename(graphroot/f"{state.name.lower()}.json")
    deletable = graphroot/f"{state.abbr.lower()}-bg-connected.json"
    os.remove(deletable)

    for geometry in ["vtd", "bg"]:
        # Create the graph path; if the path exists and the state is not in the
        # list of exceptions, keep the file; otherwise, delete it.
        graphpath = graphroot/f"{state.abbr.lower()}-{geometry}-connected.json"
